// E5A4: authoritative speculative acceptance + performance harness.
//
// Real pipeline only: target hidden taps -> DFlash2 ring (propose_cached) ->
// target H6 proposals -> serial reference verification or explicit opt-in
// transaction-backed continuation verification -> accept/reject compare ->
// ring commit of target-confirmed rows.
//
// Donor pending-anchor protocol: sample one seed from prefill logits without
// decoding it. Per read, propose with that pending token at the cache tail,
// decode/commit the anchor, then decode/commit only matching proposals. The
// first replacement or full-accept bonus is emitted and left pending. Thus each
// read emits and processes accepted+1 tokens while exactly one final token is
// absent from target/ring state. Rejected proposals never remain in target
// state and their speculative taps never reach the draft ring.
// DFlash2 ON must reproduce the DFlash2-OFF greedy stream exactly
// (paired control context, same prefix, same token count).
//
// Modes (NINFER_E5A4_MODE) include "accept" (default), "h6sweep", and the
// focused "transactionqual" gate.
#include "scoped_environment.h"
#include "exl3/text_model.h"
#include "exl3/device_prefix_cache.h"
#include "exl3/branch_reference.h"
#include "exl3/exact_outer_reference.h"
#include "exl3/fast_device_round.h"
#include "exl3/hierarchical_exact.h"
#include "exl3/turboangle_host.h"
#include "exl3/turboangle_l1.h"
#include "exl3/vericache_request.h"
#include "exl3/vericache_serving_coordinator.h"
#include "exl3/dflash2_execution.h"
#include "exl3/device_horizon_cost_policy.h"
#include "exl3/vericache_queue.h"
#include <bit>
#include <nlohmann/json.hpp>
#include <latch>
#include <thread>
#include <cuda_profiler_api.h>
#include "exl3/dflash2_draft.h"
#include "exl3/full_attention_layer.h"
#include "exl3/linear_cuda.h"
#include "exl3/linear_workspace_requirements.h"
#include "exl3/packed_projection_dispatch.h"
#include "exl3/leased_projection.h"
#include "exl3/reconstruction_stream.h"
#include "exl3/reconstruction_config.h"
#include "exl3/gdn_layer.h"
#include "exl3/oscar_runtime.h"
#include "exl3/shared_conditioning_scope.h"
#include "exl3/engine_greedy_packet_batch.h"
#include "exl3/prefill_projection_chain_graph.h"
#include "exl3/prefill_attention_chain_graph.h"
#include "core/nvtx_range.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>
#include <nvtx3/nvToolsExt.h>
#include <Windows.h>
#include <psapi.h>
#include "exl3/host_resident_set.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using ninfer::exl3::Exl3TextModel;
using ninfer::exl3::Exl3TextContext;
using ninfer::exl3::Exl3Dflash2DraftModel;
using DraftPositionConfidence = Exl3Dflash2DraftModel::DraftPositionConfidence;
using ninfer::exl3::Exl3CudaLinearWorkspace;
using ninfer::exl3::Exl3TargetProjectionPhase;
using ninfer::exl3::Exl3TargetProjectionTiming;
using ninfer::exl3::Exl3TargetProjectionTimingRecord;
using ninfer::exl3::Exl3TargetProjectionTopology;
using ninfer::exl3::exl3_fast_fused_flash_attention_calls_for_test;
using ninfer::exl3::exl3_fast_whole_context_fused_attention_calls_for_test;
using ninfer::exl3::exl3_fast_prefill_tiled_attention_calls_for_test;
using ninfer::exl3::exl3_fast_prefill_rows4_attention_calls_for_test;
using ninfer::exl3::exl3_fast_prefill_rows8_attention_calls_for_test;
using ninfer::exl3::exl3_fast_prefill_wmma_attention_calls_for_test;
using ninfer::exl3::exl3_fast_prefill_wmma32_attention_calls_for_test;
using ninfer::exl3::exl3_fast_prefill_rows2_attention_calls_for_test;
using ninfer::exl3::exl3_fast_online_decode_attention_calls_for_test;
using ninfer::exl3::exl3_fast_cublas_attention_calls_for_test;
using ninfer::exl3::target_projection_operator_name;
using ninfer::exl3::target_projection_phase_name;
using ninfer::exl3::target_projection_topology_name;

constexpr int kHidden = 5120;
constexpr int kVocab = 248320;
constexpr int kTapCount = 5;
constexpr int kMaskToken = 248070;
constexpr std::array<int, kTapCount> kTapLayers = {5, 19, 33, 47, 61};

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

std::string env(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
}

int env_int(const char* name, int fallback) {
    const std::string value = env(name);
    if (value.empty()) return fallback;
    return std::stoi(value);
}

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

#include "test_exl3_ttft.h"

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        cuda_check(cudaMalloc(&ptr_, bytes_), "E5A4 device alloc");
    }
    ~DeviceBuffer() { if (ptr_ != nullptr) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* get() const noexcept { return ptr_; }
    std::size_t bytes() const noexcept { return bytes_; }
private:
    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
};

int argmax(const std::vector<float>& logits) {
    require(!logits.empty(), "argmax on empty logits");
    int best = 0;
    for (std::size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > logits[best]) best = static_cast<int>(i);
    }
    return best;
}

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(bits & 0x8000U)) << 16U;
    const std::uint32_t exponent = (bits >> 10U) & 31U;
    const std::uint32_t fraction = bits & 1023U;
    std::uint32_t value = sign;
    if (exponent == 0U) {
        if (fraction != 0U) {
            std::uint32_t mantissa = fraction;
            int shift = 0;
            while ((mantissa & 1024U) == 0U) { mantissa <<= 1U; ++shift; }
            value |= static_cast<std::uint32_t>(127 - 14 - shift) << 23U;
            value |= (mantissa & 1023U) << 13U;
        }
    } else if (exponent == 31U) {
        value |= 0x7f800000U | (fraction << 13U);
    } else {
        value |= (exponent + 112U) << 23U | (fraction << 13U);
    }
    float result = 0.0F;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

double median(std::vector<double> values) {
    require(!values.empty(), "median of empty");
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

std::vector<std::int64_t> load_ids(const std::string& path) {
    std::ifstream in(path);
    require(in.good(), "E5A4 cannot open prompt file: " + path);
    std::vector<std::int64_t> ids;
    std::int64_t value = 0;
    while (in >> value) ids.push_back(value);
    require(!ids.empty(), "E5A4 prompt file empty: " + path);
    return ids;
}

std::int64_t sample_target(Exl3TextContext& ctx, cudaStream_t stream = nullptr) {
    if (ninfer::exl3::exl3_device_greedy_enabled())
        return ctx.greedy_packet(false, stream).decisions[0].token;
    const auto token = static_cast<std::int64_t>(argmax(ctx.logits_host(stream)));
    require(token >= 0 && token < kVocab, "sampled target token outside vocabulary");
    return token;
}

struct TapStage {
    std::vector<std::unique_ptr<DeviceBuffer>> bulk;
    std::vector<std::uint16_t*> bulk_ptrs;
    std::vector<std::unique_ptr<DeviceBuffer>> rows;
    std::vector<std::uint16_t*> row_ptrs;
};

struct DraftPrefillOverlapRoute {
    static constexpr int capacity = 1024;
    static constexpr std::size_t active_bytes =
        static_cast<std::size_t>(kTapCount) * capacity * kHidden * sizeof(std::uint16_t);
    cudaStream_t stream = nullptr;
    cudaEvent_t ready = nullptr;
    cudaEvent_t done = nullptr;
    std::array<std::unique_ptr<DeviceBuffer>, kTapCount> storage;
    std::array<std::uint16_t*, kTapCount> active{};
    double join_us = 0.0;
    bool completed = false;
    bool guards = false;

    DraftPrefillOverlapRoute() {
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "create route prefill overlap stream");
        cuda_check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming),
                   "create route prefill ready event");
        cuda_check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming),
                   "create route prefill done event");
        const std::size_t values = static_cast<std::size_t>(capacity) * kHidden + 2;
        for (int tap = 0; tap < kTapCount; ++tap) {
            storage[tap] = std::make_unique<DeviceBuffer>(values * sizeof(std::uint16_t));
            active[tap] = static_cast<std::uint16_t*>(storage[tap]->get()) + 1;
            cuda_check(cudaMemsetAsync(storage[tap]->get(), 0x5a,
                                       storage[tap]->bytes(), nullptr),
                       "poison route prefill overlap slab");
        }
    }

    ~DraftPrefillOverlapRoute() {
        if (done) cudaEventDestroy(done);
        if (ready) cudaEventDestroy(ready);
        if (stream) cudaStreamDestroy(stream);
    }

    void capture(Exl3TextContext& context, int rows) {
        require(rows >= 1 && rows <= capacity && context.captured_tap_rows() == rows,
                "route prefill overlap tap extent mismatch");
        for (int tap = 0; tap < kTapCount; ++tap)
            context.copy_tap_rows_to_device(kTapLayers[static_cast<std::size_t>(tap)],
                                            0, active[tap], rows, nullptr);
    }

    void submit(Exl3Dflash2DraftModel& draft, int rows, int absolute_position) {
        for (int first = 0; first < rows; first += 16) {
            const int count = std::min(16, rows - first);
            std::array<const std::uint16_t*, kTapCount> offset{};
            for (int tap = 0; tap < kTapCount; ++tap)
                offset[tap] = active[tap] + static_cast<std::size_t>(first) * kHidden;
            draft.commit_prefill_block(offset.data(), count,
                                       absolute_position + first, stream);
        }
    }

    void launch(Exl3TextContext& context, Exl3Dflash2DraftModel& draft,
                int rows, int absolute_position) {
        capture(context, rows);
        cuda_check(cudaEventRecord(ready, nullptr),
                   "record route prefill slab ready");
        cuda_check(cudaStreamWaitEvent(stream, ready, 0),
                   "wait route prefill slab ready");
        submit(draft, rows, absolute_position);
        cuda_check(cudaEventRecord(done, stream),
                   "record route prefill draft done");
    }

    void join() {
        const auto start = std::chrono::steady_clock::now();
        cuda_check(cudaEventSynchronize(done), "join route prefill draft work");
        join_us += std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
    }

    void stage_serial(Exl3TextContext& context, Exl3Dflash2DraftModel& draft,
                      int rows, int absolute_position) {
        launch(context, draft, rows, absolute_position);
        join();
    }

    bool guards_ok() const {
        for (int tap = 0; tap < kTapCount; ++tap) {
            std::array<std::uint16_t, 2> values{};
            const auto* base = static_cast<const std::uint16_t*>(storage[tap]->get());
            cuda_check(cudaMemcpy(&values[0], base, sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "read route prefill prefix guard");
            cuda_check(cudaMemcpy(&values[1], base + 1 + capacity * kHidden,
                                  sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                       "read route prefill suffix guard");
            if (values[0] != 0x5a5a || values[1] != 0x5a5a) return false;
        }
        return true;
    }
};

struct ContinuationGraphStreamGuard {
    cudaStream_t stream = nullptr;
    explicit ContinuationGraphStreamGuard(bool enabled) {
        if (enabled)
            cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),
                       "create fixed-B8 continuation graph stream");
    }
    ~ContinuationGraphStreamGuard() {
        if (stream) cudaStreamDestroy(stream);
    }
};

void commit_latest_row(Exl3TextContext& ctx, Exl3Dflash2DraftModel& draft,
                       TapStage& stage, int abs_pos);

// Tap capture holds ONLY the latest forward rows (overwritten per forward),
// so every ingest forward commits its rows immediately (positional slots make
// incremental commit identical to any bulk order).
void commit_current_rows(Exl3TextContext& ctx, Exl3Dflash2DraftModel& draft,
                         TapStage& stage, int rows, int abs_pos0) {
    require(rows >= 1 && rows <= 1024, "E5A4 commit rows outside forward capacity");
    // Keep the draft's qualified sixteen-row interface and scratch capacity.
    for (int first = 0; first < rows; first += 16) {
        const int count = std::min(16, rows - first);
        for (int t = 0; t < kTapCount; ++t)
            ctx.copy_tap_rows_to_device(kTapLayers[static_cast<std::size_t>(t)], first,
                static_cast<std::uint16_t*>(stage.bulk[t]->get()), count);
        draft.commit_prefill_block(const_cast<const std::uint16_t**>(stage.bulk_ptrs.data()),
                                  count, abs_pos0 + first);
    }
}

struct CommitSink {
    Exl3Dflash2DraftModel* draft = nullptr;
    TapStage* stage = nullptr;
    DraftPrefillOverlapRoute* overlap = nullptr;
};

constexpr int layer_major_retained_start(int total) {
    int retained_start=0;
    for (int forward=0;forward<total;) {
        const int forward_rows=forward==0?16:std::min(1024,total-forward);
        for (int first=0;first<forward_rows;first+=16) {
            const int rows=std::min(16,forward_rows-first);
            if (forward+first+rows<=total-2048)
                retained_start=forward+first+rows;
        }
        forward+=forward_rows;
    }
    return retained_start;
}
static_assert(layer_major_retained_start(4096)==2048);
static_assert(layer_major_retained_start(16384)==14336);
static_assert(layer_major_retained_start(4097)==2048); // crossing call halo

void diagnose_fast_device_prefill(Exl3TextContext& ctx,
    Exl3Dflash2DraftModel& draft,int total,bool fresh_window) {
    if (env("NINFER_E5A4_MODE") != "fastdevicetxn" ||
        env("NINFER_EXL3_TEST_FAST_DEVICE_REAL_DFLASH") != "1" ||
        env("NINFER_EXL3_TEST_FAST_DEVICE_ROUND_STATE") != "1") return;
    const auto status=draft.fresh_prefill_status();
    const auto submitted=status.submitted_rows?status.submitted_rows:total;
    const auto encoded=status.submitted_rows?status.encoded_rows:total;
    require(!status.active && !status.failed && submitted==total &&
            encoded+status.skipped_rows==submitted,
            "real draft prefill diagnostic coverage");
    const auto digest=draft.ring_digest();
    const auto width=std::stoi(env("NINFER_EXL3_TEST_FAST_DEVICE_REAL_WIDTH"));
    std::array<std::int64_t,8> block{};
    block.fill(248070);
    block[0]=ninfer::exl3::exl3_branch_greedy(ctx);
    const auto proposed=draft.propose_cached_view(
        std::span<const std::int64_t>(block).first(width),total,
        ctx.target_embedding(),ctx.target_lm_head_weights(),
        ctx.target_lm_head_metadata(),248070);
    require(proposed.size()==static_cast<std::size_t>(width-1),
            "real draft prefill diagnostic first proposal extent");
    std::cout<<"FAST_DEVICE_REAL_DFLASH_PREFILL_EQ window="
             <<(fresh_window?1:0)
             <<" submitted="<<submitted<<" encoded="<<encoded
             <<" skipped="<<status.skipped_rows
             <<" ring_base="<<draft.ring_base_abs()
             <<" ring_count="<<draft.ring_count()
             <<" ring_bytes="<<draft.ring_bytes()
             <<" target_hash="<<ctx.export_exact_host_state()->represented_payload_hash_for_test();
    for(std::size_t i=0;i<digest.size();++i)
        std::cout<<" ring_digest"<<i<<'='<<digest[i];
    std::cout<<" first_proposal="<<block[0];
    for(const auto token:proposed)std::cout<<','<<token;
    std::cout<<'\n';
    const auto ring_path=env("NINFER_EXL3_TEST_FAST_DEVICE_PREFILL_RING_OUT");
    require(!ring_path.empty(),"real draft prefill ring output path");
    const auto represented=draft.export_host_ring(nullptr,false);
    const auto written=represented->write_represented_payload_for_test(ring_path);
    require(written==sizeof(std::uint64_t)+sizeof(long long)+sizeof(int)+
            static_cast<std::size_t>(draft.ring_count())*5*2*2048,
            "real draft complete represented ring bytes");
    std::cout<<"FAST_DEVICE_REAL_DFLASH_RING_DUMP bytes="<<written
             <<" base="<<represented->base()
             <<" count="<<represented->count()<<'\n';
}

// Preserve the qualified initial16 prefill. The reference suffix uses teacher
// M1 decodes; the explicit chunk8 option uses native append_prefill, retaining
// continuation/M1 layer topology and only final-row logits. Commit every
// captured row to the draft ring in its original absolute order.
double ingest_prefix(Exl3TextContext& ctx, const std::vector<std::int64_t>& prefix,
                     CommitSink sink) {
    const auto chunk_setting = env("NINFER_EXL3_PREFILL_CHUNK");
    require(chunk_setting.empty() || chunk_setting == "1" || chunk_setting == "8" || chunk_setting == "16" || chunk_setting == "32" || chunk_setting == "64" || chunk_setting == "128" || chunk_setting == "256" || chunk_setting == "512" || chunk_setting == "1024",
            "prefill chunk must be empty,1,8,16,32,64 or128,256,512,1024");
    const bool wide = chunk_setting == "16" || chunk_setting == "32" || chunk_setting == "64" || chunk_setting == "128" || chunk_setting == "256" || chunk_setting == "512" || chunk_setting == "1024";
    const int wide_rows = chunk_setting == "1024" ? 1024 : chunk_setting == "512" ? 512 : chunk_setting == "256" ? 256 : chunk_setting == "128" ? 128 : chunk_setting == "64" ? 64 : chunk_setting == "32" ? 32 : 16;
    const bool chunked = chunk_setting == "8" || wide;
    const int total = static_cast<int>(prefix.size());
    require(total >= 32, "E5A4 prefix too short for a window plus anchor");
    const auto t0 = std::chrono::steady_clock::now();
    const bool fresh_window = sink.draft && env("NINFER_DFLASH2_PREFILL_WINDOW") == "1";
    const bool layer_major = env("NINFER_EXL3_FAST_LAYER_MAJOR_PREFILL") == "1";
    require(!fresh_window || chunked, "fresh draft prefill window requires chunked suffix");
    require(!layer_major || !sink.draft ||
            (sink.stage && fresh_window && wide_rows==1024 &&
             (total==4096 || total==16384) && !sink.overlap),
            "layer-major draft bridge requires scoped 4K/16K wide1024");
    if (sink.overlap) {
        require(sink.draft && sink.stage && fresh_window && wide_rows == 1024 &&
                    total == 4096 && env("NINFER_DFLASH2_PREFILL_BATCH") == "1" &&
                    env("NINFER_EXL3_WIDE_PREFILL") == "1" &&
                    env("NINFER_EXL3_PREFILL_WIDE1024") == "1",
                "route prefill overlap requires qualified eager 4K wide1024 fresh route");
        auto& route = *sink.overlap;
        try {
            sink.draft->begin_fresh_prefill(0, total, route.stream);
            TtftTrace::mark("initial16_begin", ctx.position());
            ctx.prefill(std::span<const std::int64_t>(prefix.data(), 16));
            require(ctx.position() == 16, "route overlap position after initial16");
            route.stage_serial(ctx, *sink.draft, 16, 0);
            TtftTrace::mark("initial16_complete", ctx.position(), -1, true);
            TtftTrace::mark("chunk1024_suffix_begin", ctx.position());
            ctx.append_prefill_wide(std::span<const std::int64_t>(prefix.data() + 16, 1024));
            route.stage_serial(ctx, *sink.draft, 1024, 16);
            ctx.append_prefill_wide(std::span<const std::int64_t>(prefix.data() + 1040, 1024));
            route.stage_serial(ctx, *sink.draft, 1024, 1040);
            ctx.append_prefill_wide(std::span<const std::int64_t>(prefix.data() + 2064, 1024));
            route.launch(ctx, *sink.draft, 1024, 2064);
            ctx.append_prefill_wide(std::span<const std::int64_t>(prefix.data() + 3088, 1008));
            route.join();
            route.stage_serial(ctx, *sink.draft, 1008, 3088);
            require(ctx.position() == total, "route overlap position after ingest");
            sink.draft->finish_fresh_prefill(route.stream);
            cuda_check(cudaEventRecord(route.done, route.stream),
                       "record route prefill final completion");
            route.join();
            cuda_check(cudaStreamSynchronize(nullptr),
                       "complete route prefill target work");
            route.guards = route.guards_ok();
            require(route.guards, "route prefill overlap slab guard failure");
            route.completed = true;
            TtftTrace::mark("chunk1024_suffix_complete", ctx.position());
            return std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count();
        } catch (...) {
            try {
                cudaStreamSynchronize(route.stream);
                sink.draft->reset(route.stream);
            } catch (...) {}
            throw;
        }
    }
    if (fresh_window) sink.draft->begin_fresh_prefill(0, total, nullptr);
    const int c0 = layer_major ? Exl3TextContext::layer_major_initial_rows() :
        std::min(total, 16);
    TtftTrace::mark("initial16_begin", ctx.position());
    if (c0) {
        ctx.prefill(std::span<const std::int64_t>(prefix.data(), static_cast<std::size_t>(c0)));
        require(ctx.position() == c0, "E5A4 position after first prefill chunk");
        if (sink.draft != nullptr) {
            if (layer_major) sink.draft->skip_fresh_prefill_block(c0, 0);
            else commit_current_rows(ctx, *sink.draft, *sink.stage, c0, 0);
        }
    }
    TtftTrace::mark("initial16_complete", ctx.position(), -1, true);
    TtftTrace::mark(wide_rows==1024 ? "chunk1024_suffix_begin" : wide_rows==512 ? "chunk512_suffix_begin" : wide_rows==256 ? "chunk256_suffix_begin" : wide_rows==128 ? "chunk128_suffix_begin" : wide_rows==64 ? "chunk64_suffix_begin" : wide_rows==32 ? "chunk32_suffix_begin" : wide ? "chunk16_suffix_begin" : chunked ? "chunk8_suffix_begin" : "m1_suffix_begin", ctx.position());
    if (layer_major) {
        require(wide_rows == 1024 && total > 1040 && !sink.overlap &&
                (!sink.draft || (sink.stage && fresh_window &&
                  (total == 4096 || total == 16384))),
                "layer-major prefill requires target-only or scoped real-draft wide1024");
        const auto suffix=std::span<const std::int64_t>(prefix.data()+c0,total-c0);
        if (!sink.draft) ctx.append_prefill_layer_major(suffix);
        else {
            // Reproduce the original initial16 + 1024/tail forward partitions,
            // each split into <=16-row draft calls. Retain the crossing call too.
            const int retained_start=layer_major_retained_start(total);
            const int retained_rows=total-retained_start;
            require(retained_start>=c0 && retained_rows>=2048 && retained_rows<=2063,
                    "layer-major draft retained partition halo");
            const auto bytes=static_cast<std::size_t>(kTapCount)*retained_rows*
                             kHidden*sizeof(std::uint16_t);
            DeviceBuffer retained_arena(bytes);
            Exl3TextContext::RetainedTapTail retained{
                static_cast<std::uint16_t*>(retained_arena.get()),bytes,
                retained_start,retained_rows};
            try {
                ctx.append_prefill_layer_major(suffix,nullptr,&retained);
                require(ctx.position()==total,"layer-major target prefix position");
                for (int forward=c0;forward<total;) {
                    const int forward_rows=std::min(1024,total-forward);
                    for (int first=0;first<forward_rows;first+=16) {
                        const int rows=std::min(16,forward_rows-first);
                        const int absolute=forward+first;
                        if (absolute+rows<=total-2048) {
                            sink.draft->skip_fresh_prefill_block(rows,absolute);
                        } else {
                            require(absolute>=retained_start &&
                                    absolute+rows<=total,
                                    "layer-major encoded partition outside retained taps");
                            const std::uint16_t* taps[kTapCount];
                            for (int tap=0;tap<kTapCount;++tap)
                                taps[tap]=retained.device+
                                    (static_cast<std::size_t>(tap)*retained_rows+
                                     absolute-retained_start)*kHidden;
                            sink.draft->commit_prefill_block(taps,rows,absolute);
                        }
                    }
                    forward+=forward_rows;
                }
                sink.draft->finish_fresh_prefill();
                cuda_check(cudaStreamSynchronize(nullptr),
                           "complete retained draft ring before releasing tap arena");
                std::cout << "FAST_DEVICE_REAL_DFLASH_LAYER_MAJOR_TAP_TAIL"
                          << " first=" << retained_start
                          << " rows=" << retained_rows
                          << " bytes=" << bytes << '\n';
            } catch (...) {
                cudaStreamSynchronize(nullptr);
                try {sink.draft->reset();} catch (...) {}
                throw;
            }
        }
        require(ctx.position()==total,"layer-major target prefix position");
    }
    for (int i = c0; !layer_major && i < total;) {
        const int rows = chunked ? std::min(wide ? wide_rows : 8, total - i) : 1;
        if (chunked) {
            const auto tokens = std::span<const std::int64_t>(prefix.data() + i, rows);
            if (wide) ctx.append_prefill_wide(tokens);
            else ctx.append_prefill(tokens);
            if (sink.draft != nullptr)
                commit_current_rows(ctx, *sink.draft, *sink.stage, rows, i);
        } else {
            ctx.decode(prefix[static_cast<std::size_t>(i)]);
            if (sink.draft != nullptr) commit_latest_row(ctx, *sink.draft, *sink.stage, i);
        }
        i += rows;
    }
    require(ctx.position() == total, "E5A4 position after ingest");
    if (fresh_window && !layer_major) sink.draft->finish_fresh_prefill(nullptr);
    // Complete target work and ordered ring commits inside the prefill timer;
    // the subsequent seed timer measures logits retrieval and sampling only.
    cuda_check(cudaStreamSynchronize(nullptr), "E5A4 completed prefix timing");
    if(sink.draft)diagnose_fast_device_prefill(ctx,*sink.draft,total,fresh_window);
    TtftTrace::mark(layer_major ? "layer_major_suffix_complete" : wide_rows==1024 ? "chunk1024_suffix_complete" : wide_rows==512 ? "chunk512_suffix_complete" : wide_rows==256 ? "chunk256_suffix_complete" : wide_rows==128 ? "chunk128_suffix_complete" : wide_rows==64 ? "chunk64_suffix_complete" : wide_rows==32 ? "chunk32_suffix_complete" : wide ? "chunk16_suffix_complete" : chunked ? "chunk8_suffix_complete" : "m1_suffix_complete", ctx.position());
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - t0).count();
}

// Commit the single most-recent decoded row at absolute position abs_pos.
void commit_latest_row(Exl3TextContext& ctx, Exl3Dflash2DraftModel& draft,
                       TapStage& stage, int abs_pos) {
    for (int t = 0; t < kTapCount; ++t) {
        ctx.copy_tap_row_to_device(kTapLayers[static_cast<std::size_t>(t)],
                                   static_cast<std::uint16_t*>(stage.rows[t]->get()));
    }
    // No sync here either (same stream-ordering argument as above).
    draft.commit_target_block(const_cast<const std::uint16_t**>(stage.row_ptrs.data()),
                              1, abs_pos);
}

#include "test_exl3_pending_protocol.h"
#include "test_exl3_handoff_profile.h"
#include "test_exl3_transaction_round.h"
#include "test_exl3_target_graph_c2.h"
#include "test_exl3_target_graph_c2_verifier.h"
#include "test_exl3_t74_real_c2.h"
#include "test_exl3_t78_cached_prefix_c2.h"
#include "test_exl3_t79_target_only_c2.h"
#include "test_exl3_t80_fp16_serving.h"
#include "test_exl3_t82_fp16_c2_serving.h"
#include "test_exl3_t83_fp16_c8_serving.h"
#include "test_exl3_t84_serving_coordinator.h"
#include "test_exl3_t85_fp16_c8_quantum.h"
#include "test_exl3_t86_coordinator_c8_serving.h"
#include "test_exl3_t87_batch_admission.h"
#include "test_exl3_t89_resident_profile.h"
#include "test_exl3_t90_allocation_churn.h"
#include "test_exl3_t92_persistent_turnover.h"
#include "test_exl3_t95_atomic_turnover.h"
#include "test_exl3_t97_concurrent_prefix_prepare.h"
#include "test_exl3_t98_pipelined_turnover.h"
#include "test_exl3_t101_multiturn_prefix.h"
#include "test_exl3_t88_unequal_tail_fairness.h"
#include "test_exl3_t81_cached_prefix_c4_admission.h"
#include "test_exl3_target_k6_capture.h"
#include "test_exl3_projection_ordered_c2_screen.h"
#include "test_exl3_projection_route_matrix_screen.h"
#include "test_exl3_gdn_stage_oracle.h"
#include "test_exl3_target_gateup_k7_qualification.h"
#include "test_exl3_target_o_k7_capture.h"
#include "test_exl3_dflash2_topk.h"
#include "test_exl3_dflash2_ring_attention.h"
#include "test_exl3_dflash2_dense_kmajor.h"
#include "test_exl3_target_request_contexts.h"
#include "test_exl3_oscar_default_construction.h"

std::vector<std::byte> e5a4_authoritative_state_snapshot(
    Exl3TextContext& context) {
    cuda_check(cudaDeviceSynchronize(),
               "synchronize E5A4 authoritative state snapshot");
    std::vector<std::byte> result;
    const auto append = [&](const auto& values) {
        const std::uint64_t count = values.size();
        const auto* count_bytes =
            reinterpret_cast<const std::byte*>(&count);
        result.insert(result.end(), count_bytes, count_bytes + sizeof(count));
        if (values.empty()) return;
        const auto* first =
            reinterpret_cast<const std::byte*>(values.data());
        result.insert(result.end(), first,
                      first + values.size() * sizeof(values[0]));
    };
    append(std::array<int, 2>{context.position(),
                              context.device_position_host()});
    append(transaction_round_device_bits(
        context.logits_device(), kVocab,
        "download E5A4 authoritative logits"));
    for (const int layer : kTapLayers)
        append(transaction_round_last_tap_bits(context, layer));
    for (int layer = 0; layer < 64; ++layer) {
        if (layer >= 3 && (layer - 3) % 4 == 0) continue;
        append(context.gdn_state_host(layer));
        append(context.gdn_physical_conv_host(layer));
    }
    append(context.oscar_live_state_host_for_test());
    return result;
}

struct ReadRecord {
    int read = 0;
    int attempted_block = 0;
    int accepted = 0;
    int committed = 0;
    int rejpos = -1;  // mismatch position 0..k-1, or -1 for a full-block accept
    double draft_us = 0.0;
    double verify_us = 0.0;
    double round_us = 0.0;
    // Legacy field: semantic retained rows, not physical target executions.
    // Full-model executions are attempted_verification_rows + replay_rows.
    // Retained-prefix state reconstruction is accounted separately.
    int target_decodes = 0;
    std::string proposal_ids;
    std::vector<DraftPositionConfidence> position_confidence;
    std::string verification_route = "serial";
    int attempted_verification_rows = 0;
    int replay_rows = 0;
    int retained_rows = 0;
    int state_reconstruction_rows = 0;
    bool confidence_policy_active = false;
    bool confidence_policy_burn_in = false;
    bool confidence_policy_fallback = false;
    double confidence_policy_decision_us = 0.0;
    std::array<double, 3> confidence_expected_committed{};
    std::array<double, 3> confidence_efficiency{};
};

struct ConfigResult {
    std::vector<ReadRecord> reads;
    std::vector<std::int64_t> spec_tokens;
    std::vector<std::int64_t> control_tokens;
    double prefill_us = 0.0;
    double initial_seed_us = 0.0;
    double control_prefill_us = 0.0;
    double control_seed_us = 0.0;
    double resident_ttft_ms = 0.0, resident_request_ms = 0.0;
    double control_resident_ttft_ms = 0.0, control_resident_request_ms = 0.0;
    long long draft_prefill_submitted_rows = 0, draft_prefill_encoded_rows = 0,
              draft_prefill_skipped_rows = 0;
    bool draft_prefill_overlap = false;
    std::size_t draft_prefill_overlap_staging_bytes = 0;
    double draft_prefill_overlap_join_us = 0.0;
    bool draft_prefill_overlap_guards = false;
    double control_us = 0.0;
    double spec_loop_us = 0.0;
    bool correctness = false;
    std::size_t emitted_tokens = 0;
    // Legacy total_target_decodes output is this retained semantic row count.
    std::size_t processed_tokens = 0;
    double transaction_setup_us = 0.0;
    std::size_t transaction_bytes = 0;
    std::size_t attempted_verification_rows = 0;
    std::size_t replay_rows = 0;
    std::size_t retained_rows = 0;
    std::size_t state_reconstruction_rows = 0;
    bool transaction_route = false;
    bool retain_prefix_route = false;
    bool final_state_exact = true;
    std::string verification_route = "serial";
    double target_projection_timing_setup_us = 0.0;
    double target_projection_timing_resolution_us = 0.0;
    double target_projection_raw_loop_wall_us = 0.0;
    std::vector<Exl3TargetProjectionTimingRecord> target_projection_timings;
    std::uint64_t oscar_full = 0, ordinary_full = 0, gdn_oscar = 0;
    std::size_t free_end = 0, total_mem = 0;
    int output_budget = 0;
    std::size_t useful_committed = 0;
    std::size_t surplus_retained = 0;
    double spec_context_teardown_us = 0.0;
};

struct T73ConfidencePolicy {
    struct PositionModel {
        double minimum = 0.0;
        double maximum = 0.0;
        std::array<double, 3> cuts{};
        std::array<double, 4> probabilities{};
        bool present = false;
    };
    int burn_in = 0;
    std::array<double, 3> complete_cost_us{};  // B4, B6, B8
    std::array<PositionModel, 7> positions{};
};

std::vector<std::string> split_csv_fields(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    return fields;
}

T73ConfidencePolicy load_t73_confidence_policy(const std::string& path) {
    std::ifstream input(path);
    require(input.good(), "cannot open T73 confidence policy: " + path);
    std::string line;
    require(static_cast<bool>(std::getline(input, line)) &&
                line == "T73_POLICY_V1",
            "T73 confidence policy schema mismatch");
    T73ConfidencePolicy policy;
    std::array<bool, 3> costs{};
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split_csv_fields(line);
        if (fields[0] == "burn_in") {
            require(fields.size() == 2 && policy.burn_in == 0,
                    "T73 confidence policy burn-in row mismatch");
            policy.burn_in = std::stoi(fields[1]);
        } else if (fields[0] == "cost") {
            require(fields.size() == 3, "T73 confidence policy cost row mismatch");
            const int width = std::stoi(fields[1]);
            const int slot = width == 4 ? 0 : width == 6 ? 1 : width == 8 ? 2 : -1;
            require(slot >= 0 && !costs[static_cast<std::size_t>(slot)],
                    "T73 confidence policy cost width mismatch");
            policy.complete_cost_us[static_cast<std::size_t>(slot)] =
                std::stod(fields[2]);
            require(std::isfinite(policy.complete_cost_us[static_cast<std::size_t>(slot)]) &&
                        policy.complete_cost_us[static_cast<std::size_t>(slot)] > 0.0,
                    "T73 confidence policy invalid complete cost");
            costs[static_cast<std::size_t>(slot)] = true;
        } else if (fields[0] == "model") {
            require(fields.size() == 11, "T73 confidence policy model row mismatch");
            const int position = std::stoi(fields[1]);
            require(position >= 0 && position < 7 &&
                        !policy.positions[static_cast<std::size_t>(position)].present,
                    "T73 confidence policy position mismatch");
            auto& model = policy.positions[static_cast<std::size_t>(position)];
            model.minimum = std::stod(fields[2]);
            model.maximum = std::stod(fields[3]);
            for (int i = 0; i < 3; ++i) model.cuts[static_cast<std::size_t>(i)] = std::stod(fields[4 + i]);
            for (int i = 0; i < 4; ++i) model.probabilities[static_cast<std::size_t>(i)] = std::stod(fields[7 + i]);
            require(std::isfinite(model.minimum) && std::isfinite(model.maximum) &&
                        model.minimum >= 0.0 && model.minimum <= model.cuts[0] &&
                        model.cuts[0] <= model.cuts[1] &&
                        model.cuts[1] <= model.cuts[2] &&
                        model.cuts[2] <= model.maximum,
                    "T73 confidence policy feature range mismatch");
            for (int i = 0; i < 4; ++i)
                require(std::isfinite(model.probabilities[static_cast<std::size_t>(i)]) &&
                            model.probabilities[static_cast<std::size_t>(i)] > 0.0 &&
                            model.probabilities[static_cast<std::size_t>(i)] < 1.0 &&
                            (i == 0 || model.probabilities[static_cast<std::size_t>(i - 1)] <=
                                           model.probabilities[static_cast<std::size_t>(i)]),
                        "T73 confidence policy probability mismatch");
            model.present = true;
        } else {
            require(false, "T73 confidence policy unknown row");
        }
    }
    require(policy.burn_in == 8 &&
                std::all_of(costs.begin(), costs.end(), [](bool value) { return value; }) &&
                std::all_of(policy.positions.begin(), policy.positions.end(),
                            [](const auto& model) { return model.present; }),
            "T73 confidence policy incomplete");
    return policy;
}

int choose_t73_confidence_block(
    const T73ConfidencePolicy& policy,
    const std::vector<DraftPositionConfidence>& confidence,
    ReadRecord& record) {
    require(confidence.size() == 7,
            "T73 confidence policy requires a B8 proposal");
    std::array<double, 7> probabilities{};
    bool fallback = false;
    for (int position = 0; position < 7; ++position) {
        const auto& observation = confidence[static_cast<std::size_t>(position)];
        const double scale = std::max(
            {std::abs(static_cast<double>(observation.selected_edge_score)),
             std::abs(static_cast<double>(observation.runner_up_edge_score)),
             1.0e-6});
        const double feature = static_cast<double>(observation.edge_margin) / scale;
        const auto& model = policy.positions[static_cast<std::size_t>(position)];
        if (!std::isfinite(feature) || feature < model.minimum || feature > model.maximum) {
            fallback = true;
            break;
        }
        int bin = 0;
        while (bin < 3 && feature > model.cuts[static_cast<std::size_t>(bin)]) ++bin;
        probabilities[static_cast<std::size_t>(position)] =
            model.probabilities[static_cast<std::size_t>(bin)];
    }
    record.confidence_policy_fallback = fallback;
    if (fallback) return 8;
    constexpr std::array<int, 3> widths{4, 6, 8};
    int selected_slot = 2;
    double best_efficiency = -1.0;
    for (int slot = 0; slot < 3; ++slot) {
        double survival = 1.0;
        double expected = 1.0;
        for (int position = 0; position < widths[static_cast<std::size_t>(slot)] - 1;
             ++position) {
            survival *= probabilities[static_cast<std::size_t>(position)];
            expected += survival;
        }
        const double efficiency =
            expected / policy.complete_cost_us[static_cast<std::size_t>(slot)];
        record.confidence_expected_committed[static_cast<std::size_t>(slot)] = expected;
        record.confidence_efficiency[static_cast<std::size_t>(slot)] = efficiency;
        // Widths are visited from small to large; ties deliberately prefer the
        // larger verifier to retain more L2 evidence.
        if (efficiency >= best_efficiency) {
            best_efficiency = efficiency;
            selected_slot = slot;
        }
    }
    return widths[static_cast<std::size_t>(selected_slot)];
}

// Runs one (family, context, block) configuration. Reads beyond the warmup
// count are measured; warmup reads still commit tokens (stream stays exact).
ConfigResult run_config(Exl3TextModel& target, Exl3Dflash2DraftModel& draft,
                        const std::vector<std::int64_t>& prefix, int block_len,
                        int reads, int warmup, bool oscar,
                        HandoffProfile* handoff_profile = nullptr) {
    require(block_len >= 2 && block_len <= draft.block_capacity(), "E5A4 block length outside capacity");
    const int k = block_len - 1;
    ConfigResult out;
    out.output_budget = env_int("NINFER_E5A4_OUTPUT_BUDGET", 0);
    require(out.output_budget >= 0,
            "E5A4 output budget must be nonnegative");
    const std::string verifier = env("NINFER_E5A4_VERIFIER");
    require(verifier.empty() || verifier == "serial" || verifier == "transaction" ||
                verifier == "transaction_retain",
            "E5A4 VERIFIER must be serial, transaction, or transaction_retain");
    out.transaction_route =
        verifier == "transaction" || verifier == "transaction_retain";
    out.retain_prefix_route = verifier == "transaction_retain";
    out.verification_route = verifier.empty() ? "serial" : verifier;
    require(!out.transaction_route || oscar,
            "transaction verifier requires canonical OSCAR");
    const std::string continuation_graph_value =
        env("NINFER_E5A4_CONTINUATION_GRAPH_B8");
    require(continuation_graph_value.empty() || continuation_graph_value == "0" ||
                continuation_graph_value == "1",
            "fixed-B8 continuation graph flag must be 0 or 1");
    const std::string continuation_cohort_value =
        env("NINFER_OSCAR_CONTINUATION_COHORT_B8");
    require(continuation_cohort_value.empty() || continuation_cohort_value == "0" ||
                continuation_cohort_value == "1",
            "OSCAR continuation cohort B8 flag must be 0 or 1");
    const bool continuation_graph = continuation_graph_value == "1";
    const bool continuation_cohort = continuation_cohort_value == "1";
    const std::string adaptive_block_value =
        env("NINFER_E5A4_ADAPTIVE_BLOCK_LAG1");
    require(adaptive_block_value.empty() || adaptive_block_value == "0" ||
                adaptive_block_value == "1",
            "adaptive block lag1 flag must be 0 or 1");
    const bool adaptive_eager_block = adaptive_block_value == "1";
    const std::string adaptive_graph_value =
        env("NINFER_E5A4_ADAPTIVE_GRAPH_LAG1");
    require(adaptive_graph_value.empty() || adaptive_graph_value == "0" ||
                adaptive_graph_value == "1",
            "adaptive graph lag1 flag must be 0 or 1");
    const bool adaptive_graph_block = adaptive_graph_value == "1";
    const std::string confidence_forced_graph_value =
        env("NINFER_E5A4_CONFIDENCE_FORCED_GRAPH_BLOCK");
    require(confidence_forced_graph_value.empty() ||
                confidence_forced_graph_value == "0" ||
                confidence_forced_graph_value == "4" ||
                confidence_forced_graph_value == "6" ||
                confidence_forced_graph_value == "8",
            "confidence forced graph block must be 0, 4, 6, or 8");
    const int confidence_forced_graph_block =
        confidence_forced_graph_value.empty()
            ? 0
            : std::stoi(confidence_forced_graph_value);
    const bool confidence_cost_calibration =
        confidence_forced_graph_block != 0;
    const std::string confidence_policy_value =
        env("NINFER_E5A4_CONFIDENCE_POLICY");
    require(confidence_policy_value.empty() || confidence_policy_value == "0" ||
                confidence_policy_value == "1",
            "confidence policy flag must be 0 or 1");
    const bool confidence_policy_active = confidence_policy_value == "1";
    T73ConfidencePolicy confidence_policy;
    if (confidence_policy_active) {
        const std::string policy_path =
            env("NINFER_E5A4_CONFIDENCE_POLICY_FILE");
        require(!policy_path.empty(), "confidence policy requires a policy file");
        confidence_policy = load_t73_confidence_policy(policy_path);
    }
    const std::string adaptive_graph_min6_value =
        env("NINFER_E5A4_ADAPTIVE_GRAPH_MIN6");
    require(adaptive_graph_min6_value.empty() ||
                adaptive_graph_min6_value == "0" ||
                adaptive_graph_min6_value == "1",
            "adaptive graph min6 flag must be 0 or 1");
    const bool adaptive_graph_min6 = adaptive_graph_min6_value == "1";
    require(!(adaptive_eager_block && adaptive_graph_block),
            "adaptive eager and graph lag1 flags are mutually exclusive");
    require(!confidence_cost_calibration ||
                (!adaptive_eager_block && !adaptive_graph_block &&
                 env("NINFER_DFLASH2_POSITION_CONFIDENCE") == "1"),
            "confidence forced graph block requires confidence telemetry and no adaptive lag1 policy");
    require(!confidence_policy_active ||
                (!confidence_cost_calibration && !adaptive_eager_block &&
                 !adaptive_graph_block &&
                 env("NINFER_DFLASH2_POSITION_CONFIDENCE") == "1"),
            "confidence policy requires confidence telemetry and excludes forced/adaptive policies");
    require(!adaptive_graph_min6 || adaptive_graph_block,
            "adaptive graph min6 requires adaptive graph lag1");
    const bool adaptive_block = adaptive_eager_block || adaptive_graph_block;
    const std::string final_state_check_value =
        env("NINFER_E5A4_FINAL_STATE_CHECK");
    require(final_state_check_value.empty() || final_state_check_value == "0" ||
                final_state_check_value == "1",
            "final state check flag must be 0 or 1");
    const bool final_state_check = final_state_check_value == "1";
    const std::string measure_teardown_value =
        env("NINFER_E5A4_MEASURE_CONTEXT_TEARDOWN");
    require(measure_teardown_value.empty() || measure_teardown_value == "0" ||
                measure_teardown_value == "1",
            "context teardown measurement flag must be 0 or 1");
    const bool measure_teardown = measure_teardown_value == "1";
    const std::string continuation_gdn_qkvz_value =
        env("NINFER_EXL3_CONTINUATION_GRAPH_GDN_QKVZ_CONCURRENT");
    require(continuation_gdn_qkvz_value.empty() ||
                continuation_gdn_qkvz_value == "0" ||
                continuation_gdn_qkvz_value == "1",
            "fixed-B8 graph GDN QKV/Z concurrency flag must be 0 or 1");
    const bool continuation_gdn_qkvz =
        continuation_graph && continuation_gdn_qkvz_value == "1";
    const std::string prefill_overlap_value =
        env("NINFER_EXL3_DRAFT_PREFILL_OVERLAP");
    require(prefill_overlap_value.empty() || prefill_overlap_value == "0" ||
                prefill_overlap_value == "1",
            "draft prefill overlap flag must be 0 or 1");
    const bool prefill_overlap = prefill_overlap_value == "1";
    require(!adaptive_eager_block ||
                (out.retain_prefix_route && block_len == 8 &&
                 !continuation_graph && !handoff_profile &&
                 env("NINFER_EXL3_TARGET_PROJECTION_TIMING").empty()),
            "adaptive block lag1 requires uninstrumented eager transaction-retain capacity B8");
    require(!adaptive_graph_block ||
                (out.retain_prefix_route && block_len == 8 &&
                 continuation_graph && continuation_cohort &&
                 !handoff_profile &&
                 env("NINFER_EXL3_TARGET_PROJECTION_TIMING").empty()),
            "adaptive graph lag1 requires uninstrumented transaction-retain graph capacity B8");
    require(!confidence_cost_calibration ||
                (out.retain_prefix_route && block_len == 8 &&
                 continuation_graph && continuation_cohort &&
                 !handoff_profile &&
                 env("NINFER_EXL3_TARGET_PROJECTION_TIMING").empty()),
            "confidence forced graph block requires uninstrumented transaction-retain graph capacity B8");
    require(!confidence_policy_active ||
                (out.retain_prefix_route && block_len == 8 &&
                 continuation_graph && continuation_cohort &&
                 !handoff_profile &&
                 env("NINFER_EXL3_TARGET_PROJECTION_TIMING").empty()),
            "confidence policy requires uninstrumented transaction-retain graph capacity B8");
    require(!continuation_graph ||
                (out.transaction_route && block_len == 8 && oscar && !handoff_profile &&
                 env("NINFER_EXL3_TARGET_PROJECTION_TIMING").empty()),
            "fixed-B8 continuation graph requires uninstrumented OSCAR transaction B8");
    if (continuation_graph)
        out.verification_route += continuation_cohort
            ? "_graph_oscar_cohort_b8" : "_graph_oscar_sequential_b8";
    if (continuation_gdn_qkvz)
        out.verification_route += "_gdn_qkvz_concurrent";
    if (adaptive_block)
        out.verification_route += "_adaptive_block_lag1";
    if (adaptive_graph_min6)
        out.verification_route += "_min6";
    if (confidence_cost_calibration)
        out.verification_route += "_confidence_forced_b" +
            std::to_string(confidence_forced_graph_block);
    if (confidence_policy_active)
        out.verification_route += "_confidence_policy_v1";
    // Both arms use the same named nonblocking stream so graph admission is
    // the only performance variable in a matched transaction-B8 screen.
    ContinuationGraphStreamGuard continuation_graph_stream(
        out.transaction_route && block_len == 8);
    cudaStream_t verification_stream = continuation_graph_stream.stream;
    const auto resident0 = std::chrono::steady_clock::now();
    TtftTrace::scope("spec", static_cast<int>(prefix.size()));
    TtftTrace::mark("tap_stage_begin");
    TapStage stage;
    for (int t = 0; t < kTapCount; ++t) {
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(
            static_cast<std::size_t>(16) * kHidden * sizeof(std::uint16_t)));
        stage.rows.push_back(std::make_unique<DeviceBuffer>(
            static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t)));
        stage.bulk_ptrs.push_back(static_cast<std::uint16_t*>(stage.bulk.back()->get()));
        stage.row_ptrs.push_back(static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }
    constexpr std::size_t kTapStageBytes =
        static_cast<std::size_t>(kTapCount) * 17 * kHidden * sizeof(std::uint16_t);
    TtftTrace::mark("tap_stage_complete", -1, -1, true);
    if (handoff_profile)
        handoff_profile->sample_memory(
            "tap_stage", -1, &target, &draft, nullptr, kTapStageBytes);
    // ---- Spec phase (tap-capturing context). Destroyed before control so peak
    // VRAM holds only one long-context KV at a time. ----
    std::uint64_t spec_oscar_full = 0, spec_ordinary_full = 0, spec_gdn = 0;
    std::vector<std::byte> spec_final_state;
    {
        TtftTrace::mark("context_begin");
        auto spec = target.create_context(true);
        TtftTrace::mark("context_complete", spec->position(), -1, true);
        if (handoff_profile)
            handoff_profile->sample_memory(
                "spec_context_created", -1, &target, &draft, spec.get(), kTapStageBytes);
        TtftTrace::mark("oscar_attach_begin");
        if (oscar) {
            require(spec->try_enable_oscar_from_environment(), "E5A4 OSCAR enable (spec) failed");
        }
        TtftTrace::mark("oscar_attach_complete", spec->position(), -1, true);
        if (handoff_profile)
            handoff_profile->sample_memory(
                "spec_oscar_enabled", -1, &target, &draft, spec.get(), kTapStageBytes);
        TtftTrace::mark("draft_reset_begin");
        std::unique_ptr<DraftPrefillOverlapRoute> prefill_overlap_route;
        if (prefill_overlap)
            prefill_overlap_route = std::make_unique<DraftPrefillOverlapRoute>();
        draft.reset(prefill_overlap_route ? prefill_overlap_route->stream : nullptr);
        TtftTrace::mark("draft_reset_complete", spec->position(), -1, true);
        CommitSink sink{&draft, &stage, prefill_overlap_route.get()};
        HandoffProfile::Handle spec_prefill;
        if (handoff_profile)
            spec_prefill = handoff_profile->begin(-1, "spec", "prefill");
        out.prefill_us = ingest_prefix(*spec, prefix, sink);
        // Unscoped counts derive from unchanged complete ingestion; scoped counts are native.
        const auto prefill_status = draft.fresh_prefill_status();
        out.draft_prefill_submitted_rows = prefill_status.submitted_rows ? prefill_status.submitted_rows : prefix.size();
        out.draft_prefill_encoded_rows = prefill_status.submitted_rows ? prefill_status.encoded_rows : prefix.size();
        out.draft_prefill_skipped_rows = prefill_status.skipped_rows;
        if (prefill_overlap_route) {
            require(prefill_overlap_route->completed && prefill_overlap_route->guards,
                    "draft prefill overlap did not complete exact route guard");
            out.draft_prefill_overlap = true;
            out.draft_prefill_overlap_staging_bytes =
                DraftPrefillOverlapRoute::active_bytes;
            out.draft_prefill_overlap_join_us = prefill_overlap_route->join_us;
            out.draft_prefill_overlap_guards = true;
            prefill_overlap_route.reset();
        }
        if (handoff_profile) {
            handoff_profile->end(spec_prefill);
            handoff_profile->resolve();
            handoff_profile->sample_memory(
                "spec_prefix_complete", -1, &target, &draft, spec.get(), kTapStageBytes);
        }
        if (out.transaction_route) {
            TtftTrace::mark("transaction_setup_begin", spec->position());
            const auto setup0 = std::chrono::steady_clock::now();
            spec->prepare_transaction();
            spec->prepare_continuation(block_len);
            cuda_check(cudaStreamSynchronize(nullptr),
                       "synchronize transaction verifier setup");
            if (continuation_graph) {
                cuda_check(cudaDeviceSynchronize(),
                           "synchronize before fixed-B8 continuation graph setup");
                require(spec->capture_continuation_graph(verification_stream),
                        "fixed-B8 continuation graph capture failed: " +
                            spec->continuation_graph_status());
                require(spec->continuation_graph_status().find(
                            continuation_gdn_qkvz ? "sibling GDN QKV/Z projections" :
                                                   "sequential GDN QKV/Z") !=
                            std::string::npos,
                        "fixed-B8 continuation graph GDN QKV/Z route marker mismatch");
                if (adaptive_graph_block || confidence_cost_calibration ||
                    confidence_policy_active) {
                    for (const int rows : {4, 6}) {
                        if (adaptive_graph_min6 && rows == 4) continue;
                        require(spec->capture_continuation_graph_rows(
                                    rows, verification_stream),
                                "adaptive continuation B" +
                                    std::to_string(rows) +
                                    " graph capture failed: " +
                                    spec->continuation_graph_rows_status(rows));
                        require(spec->continuation_graph_rows_status(rows).find(
                                    continuation_gdn_qkvz
                                        ? "sibling GDN QKV/Z projections"
                                        : "sequential GDN QKV/Z") !=
                                    std::string::npos,
                                "adaptive continuation graph GDN QKV/Z route marker mismatch");
                    }
                }
                cuda_check(cudaStreamSynchronize(verification_stream),
                           "synchronize fixed-B8 continuation graph setup");
            }
            out.transaction_setup_us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - setup0).count();
            out.transaction_bytes = spec->transaction_bytes() + spec->continuation_bytes();
            TtftTrace::mark("transaction_setup_complete", spec->position());
            if (handoff_profile)
                handoff_profile->sample_memory(
                    "spec_prepared", -1, &target, &draft, spec.get(), kTapStageBytes);
        }
        if (spec->target_projection_timing_enabled()) {
            require(out.transaction_route,
                    "target projection timing requires transaction verification");
            const auto timing_setup0 = std::chrono::steady_clock::now();
            spec->prepare_target_projection_timing();
            out.target_projection_timing_setup_us =
                std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - timing_setup0).count();
        }
        const int tail = std::min(static_cast<int>(prefix.size()),
                                  Exl3Dflash2DraftModel::ring_keep());
        require(draft.ring_count() == tail, "E5A4 ring count after ingest");
        require(draft.ring_base_abs() == static_cast<long long>(prefix.size() - tail),
                "E5A4 ring base after ingest");
        const auto seed0 = std::chrono::steady_clock::now();
        TtftTrace::mark("seed_begin", spec->position());
        std::int64_t pending = sample_target(*spec);
        const auto seed1 = std::chrono::steady_clock::now();
        out.initial_seed_us = std::chrono::duration<double, std::micro>(seed1 - seed0).count();
        out.spec_tokens.reserve(1u + static_cast<std::size_t>(reads) *
                                      static_cast<std::size_t>(k + 1));
        out.spec_tokens.push_back(pending);  // emitted, intentionally not decoded
        out.resident_ttft_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - resident0).count();
        TtftTrace::mark("first_token", spec->position(), pending);
        TtftTrace::mark("graph_eager_no_setup", spec->position());
        int abs_pos = static_cast<int>(prefix.size());  // target/ring tail
    const auto loop0 = std::chrono::steady_clock::now();
    int previous_accepted = 0;
    for (int r = 0;
         r < reads &&
         (out.output_budget == 0 ||
          abs_pos - static_cast<int>(prefix.size()) < out.output_budget);
         ++r) {
        int round_block_len = confidence_cost_calibration
            ? confidence_forced_graph_block
            : (adaptive_block
                   ? (adaptive_graph_min6
                          ? (previous_accepted <= 4 ? 6 : 8)
                          : (previous_accepted <= 1 ? 4 :
                             previous_accepted <= 4 ? 6 : 8))
                   : block_len);
        int round_k = round_block_len - 1;
        const int draft_block_len =
            (confidence_cost_calibration || confidence_policy_active)
            ? block_len
            : round_block_len;
        const int draft_k = draft_block_len - 1;
        const auto round0 = std::chrono::steady_clock::now();
        HandoffProfile::Handle full_round;
        if (handoff_profile)
            full_round = handoff_profile->begin(r, "round", "full_round");
        ReadRecord rec;
        rec.read = r;
        std::vector<std::int64_t> block(
            static_cast<std::size_t>(draft_block_len), kMaskToken);
        block[0] = pending;
        const auto d0 = std::chrono::steady_clock::now();
        HandoffProfile::Handle proposal_stage;
        if (handoff_profile)
            proposal_stage = handoff_profile->begin(
                r, "proposal", "propose_cached");
        std::vector<std::int64_t> proposals = draft.propose_cached(
            block, abs_pos, spec->target_embedding(),
            spec->target_lm_head_weights(), spec->target_lm_head_metadata(), kMaskToken);
        const auto d1 = std::chrono::steady_clock::now();
        if (handoff_profile) handoff_profile->end(proposal_stage);
        rec.draft_us = std::chrono::duration<double, std::micro>(d1 - d0).count();
        require(proposals.size() == static_cast<std::size_t>(draft_k),
                "E5A4 proposal count mismatch");
        rec.position_confidence = draft.last_position_confidence_for_test();
        require(rec.position_confidence.empty() ||
                    rec.position_confidence.size() == proposals.size(),
                "E5A4 position confidence count mismatch");
        for (const auto& confidence : rec.position_confidence) {
            require(std::isfinite(confidence.selected_edge_score) &&
                        std::isfinite(confidence.runner_up_edge_score) &&
                        std::isfinite(confidence.edge_margin) &&
                        std::isfinite(confidence.selected_unary_score) &&
                        std::isfinite(confidence.best_unary_score) &&
                        std::isfinite(confidence.runner_up_unary_score) &&
                        std::isfinite(confidence.unary_margin) &&
                        confidence.edge_margin >= 0.0F &&
                        confidence.unary_margin >= 0.0F &&
                        confidence.selected_candidate_rank >= 0 &&
                        confidence.selected_candidate_rank < 16,
                    "E5A4 non-finite or invalid position confidence");
        }
        if (confidence_policy_active) {
            rec.confidence_policy_active = true;
            if (r < confidence_policy.burn_in) {
                rec.confidence_policy_burn_in = true;
                round_block_len = 8;
            } else {
                const auto decision_begin = std::chrono::steady_clock::now();
                round_block_len = choose_t73_confidence_block(
                    confidence_policy, rec.position_confidence, rec);
                rec.confidence_policy_decision_us =
                    std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - decision_begin).count();
            }
            round_k = round_block_len - 1;
        }
        rec.attempted_block = round_block_len;
        if (r == 0 && !env("NINFER_E5A4_TOPK_CHECK_OUT").empty())
            check_real_topk(draft, draft_k, env("NINFER_E5A4_TOPK_CHECK_OUT"));
        {
            std::ostringstream ids;
            for (std::size_t i = 0; i < proposals.size(); ++i) {
                if (i != 0) ids << '|';
                ids << proposals[i];
            }
            rec.proposal_ids = ids.str();
        }
        // Explicit untimed diagnostic run only. Ordinary performance runs leave
        // this unset; the copy/file write is intentionally outside draft_us.
        const auto capture_path = env("NINFER_E5A5H_CAPTURE");
        if (r == 0 && !capture_path.empty()) {
            std::vector<std::uint16_t> head_input(
                static_cast<std::size_t>(round_k) * kHidden);
            cuda_check(cudaMemcpy(head_input.data(), draft.last_head_input_device_for_test(),
                head_input.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "capture H6 input");
            std::ofstream capture(capture_path, std::ios::binary);
            capture.write(reinterpret_cast<const char*>(head_input.data()),
                          static_cast<std::streamsize>(head_input.size() * sizeof(std::uint16_t)));
            require(capture.good(), "cannot write H6 activation capture");
        }
        for (const auto token : proposals) {
            require(token >= 0 && token < kVocab, "E5A4 proposal outside vocabulary");
            require(std::isfinite(static_cast<double>(token)), "E5A4 non-finite proposal");
        }
        if (confidence_cost_calibration || confidence_policy_active)
            proposals.resize(static_cast<std::size_t>(round_k));
        if (handoff_profile && r == 0)
            handoff_profile->sample_memory(
                "proposal_complete", r, &target, &draft, spec.get(), kTapStageBytes);
        const auto v0 = std::chrono::steady_clock::now();
        PendingRoundResult round;
        if (out.transaction_route) {
            if (spec->target_projection_timing_enabled())
                spec->begin_target_projection_timing_round(r, verification_stream);
            const TransactionRoundResult transaction = verify_pending_round_transactional(
                *spec, draft, stage, pending, proposals, abs_pos,
                verification_stream, -1, 0x7e00u,
                out.retain_prefix_route
                    ? TransactionRepairMode::retain_prefix
                    : TransactionRepairMode::rollback_replay,
                -1, handoff_profile, r);
            round.emitted = transaction.emitted;
            round.pending = transaction.pending;
            round.accepted = transaction.accepted;
            round.processed = transaction.retained_rows;
            round.rejection_index = transaction.rejection_index;
            rec.verification_route = out.verification_route;
            rec.attempted_verification_rows = transaction.attempted_rows;
            rec.replay_rows = transaction.replay_rows;
            rec.retained_rows = transaction.retained_rows;
            rec.state_reconstruction_rows = transaction.state_reconstruction_rows;
            require(!out.retain_prefix_route ||
                        (rec.replay_rows == 0 &&
                         rec.state_reconstruction_rows ==
                             (transaction.rejection_index < 0
                                  ? 0
                                  : rec.retained_rows)),
                    "retained-prefix verifier per-round work accounting mismatch");
            abs_pos += transaction.retained_rows;
        } else {
            round = verify_pending_round(
                pending, proposals,
                [&]() { return sample_target(*spec); },
                [&](std::int64_t token) {
                    spec->decode(token);
                    commit_latest_row(*spec, draft, stage, abs_pos);
                    ++abs_pos;
                });
            rec.attempted_verification_rows = round.processed;
            rec.retained_rows = round.processed;
        }
        const auto v1 = std::chrono::steady_clock::now();
        rec.verify_us = std::chrono::duration<double, std::micro>(v1 - v0).count();
        rec.accepted = round.accepted;
        previous_accepted = round.accepted;
        rec.committed = round.processed;
        rec.target_decodes = round.processed;
        rec.rejpos = round.rejection_index;
        pending = round.pending;
        out.spec_tokens.insert(out.spec_tokens.end(), round.emitted.begin(), round.emitted.end());
        require(round.emitted.size() == static_cast<std::size_t>(round.accepted + 1) &&
                    round.processed == round.accepted + 1,
                "pending round accounting mismatch");
        require(spec->position() == abs_pos &&
                    draft.ring_base_abs() + draft.ring_count() == abs_pos &&
                    abs_pos == static_cast<int>(prefix.size() + out.spec_tokens.size() - 1),
                "pending round target/ring/emitted position mismatch");
        rec.round_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - round0).count();
        if (spec->target_projection_timing_enabled()) {
            const auto resolution0 = std::chrono::steady_clock::now();
            auto timings =
                spec->finish_target_projection_timing_round_after_synchronize();
            const std::size_t expected = static_cast<std::size_t>(
                Exl3TargetProjectionTiming::kGroupsPerPass *
                (1 + rec.replay_rows));
            require(timings.size() == expected,
                    "target projection timing group count mismatch");
            const auto attempt_groups = std::count_if(
                timings.begin(), timings.end(), [](const auto& timing) {
                    return timing.phase == Exl3TargetProjectionPhase::attempt;
                });
            const auto replay_groups = std::count_if(
                timings.begin(), timings.end(), [](const auto& timing) {
                    return timing.phase == Exl3TargetProjectionPhase::replay;
                });
            require(attempt_groups == Exl3TargetProjectionTiming::kGroupsPerPass &&
                        replay_groups == Exl3TargetProjectionTiming::kGroupsPerPass *
                            rec.replay_rows,
                    "target projection timing phase count mismatch");
            const bool target_kv_small_m_requested =
                env("NINFER_EXL3_TARGET_KV_SMALL_M") == "1";
            const bool target_gateup_k5_small_m_requested =
                env("NINFER_EXL3_TARGET_GATEUP_K5_SMALL_M") == "1";
            const bool target_k5_small_m_batch_requested =
                env("NINFER_EXL3_TARGET_K5_SMALL_M_BATCH") == "1";
            const bool target_gateup_small_m_requested =
                env("NINFER_EXL3_TARGET_GATEUP_SMALL_M") == "1";
            const bool target_gateup_k7_small_m_requested =
                env("NINFER_EXL3_TARGET_GATEUP_K7_SMALL_M") == "1";
            const bool target_down_small_m_requested =
                env("NINFER_EXL3_TARGET_DOWN_SMALL_M") == "1";
            const bool target_down_k7_small_m_requested =
                env("NINFER_EXL3_TARGET_DOWN_K7_SMALL_M") == "1";
            const bool target_o_k6_small_m_requested =
                env("NINFER_EXL3_TARGET_O_K6_SMALL_M") == "1";
            const bool target_o_k7_small_m_requested =
                env("NINFER_EXL3_TARGET_O_K7_SMALL_M") == "1";
            const bool target_qkv_k6_small_m_requested =
                env("NINFER_EXL3_TARGET_QKV_K6_SMALL_M") == "1";
            const bool target_q_k6_small_m_requested =
                env("NINFER_EXL3_TARGET_Q_K6_SMALL_M") == "1";
            const bool target_z_k6_small_m_requested =
                env("NINFER_EXL3_TARGET_Z_K6_SMALL_M") == "1";
            for (const auto& timing : timings) {
                if (timing.layer < 0) continue;
                const bool attempt =
                    timing.phase == Exl3TargetProjectionPhase::attempt;
                const bool gateup_candidate_eligible =
                    attempt && ((target_gateup_small_m_requested && timing.K == 6) ||
                                (target_gateup_k7_small_m_requested && timing.K == 7) ||
                                (target_gateup_k5_small_m_requested && timing.K == 5)) &&
                    (timing.operation ==
                         ninfer::exl3::Exl3TargetProjectionOperator::gate ||
                     timing.operation ==
                         ninfer::exl3::Exl3TargetProjectionOperator::up) &&
                    timing.in_features == 5120 && timing.out_features == 17408;
                const bool down_candidate_eligible =
                    attempt && target_down_small_m_requested &&
                    timing.operation ==
                        ninfer::exl3::Exl3TargetProjectionOperator::down &&
                    timing.K == 6 && timing.in_features == 17408 &&
                    timing.out_features == 5120;
                const bool o_k7_candidate_eligible =
                    attempt && target_o_k7_small_m_requested &&
                    timing.operation ==
                        ninfer::exl3::Exl3TargetProjectionOperator::o &&
                    timing.K == 7 && timing.in_features == 6144 &&
                    timing.out_features == 5120;
                const bool o_k6_candidate_eligible =
                    attempt && target_o_k6_small_m_requested &&
                    timing.operation ==
                        ninfer::exl3::Exl3TargetProjectionOperator::o &&
                    timing.K == 6 && timing.in_features == 6144 &&
                    timing.out_features == 5120;
                const bool down_k7_candidate_eligible =
                    attempt && target_down_k7_small_m_requested &&
                    timing.operation ==
                        ninfer::exl3::Exl3TargetProjectionOperator::down &&
                    timing.K == 7 && timing.in_features == 17408 &&
                    timing.out_features == 5120;
                const bool z_k6_candidate_eligible =
                    attempt && target_z_k6_small_m_requested &&
                    timing.operation == ninfer::exl3::Exl3TargetProjectionOperator::z &&
                    timing.K == 6 && timing.in_features == 5120 &&
                    timing.out_features == 6144;
                const bool qkv_k6_candidate_eligible =
                    attempt && target_qkv_k6_small_m_requested &&
                    timing.operation == ninfer::exl3::Exl3TargetProjectionOperator::qkv &&
                    timing.K == 6 && timing.in_features == 5120 &&
                    timing.out_features == 10240 && timing.rows >= 2 && timing.rows <= 8;
                const bool q_k6_candidate_eligible =
                    attempt && target_q_k6_small_m_requested &&
                    timing.operation == ninfer::exl3::Exl3TargetProjectionOperator::q &&
                    timing.K == 6 && timing.in_features == 5120 &&
                    timing.out_features == 12288 && timing.rows >= 2 && timing.rows <= 8;
                const bool kv_candidate_eligible = attempt && target_kv_small_m_requested &&
                    (timing.operation == ninfer::exl3::Exl3TargetProjectionOperator::k || timing.operation == ninfer::exl3::Exl3TargetProjectionOperator::v) &&
                    (timing.K == 7 || timing.K == 8) && timing.in_features == 5120 && timing.out_features == 1024;
                const bool k5_batch_candidate_eligible = attempt &&
                    target_k5_small_m_batch_requested && timing.K == 5 &&
                    ((timing.operation == ninfer::exl3::Exl3TargetProjectionOperator::q &&
                      timing.in_features == 5120 && timing.out_features == 12288) ||
                     (timing.operation == ninfer::exl3::Exl3TargetProjectionOperator::qkv &&
                      timing.in_features == 5120 && timing.out_features == 10240) ||
                     (timing.operation == ninfer::exl3::Exl3TargetProjectionOperator::z &&
                      timing.in_features == 5120 && timing.out_features == 6144) ||
                     ((timing.operation == ninfer::exl3::Exl3TargetProjectionOperator::k ||
                       timing.operation == ninfer::exl3::Exl3TargetProjectionOperator::v) &&
                      timing.in_features == 5120 && timing.out_features == 1024) ||
                     (timing.operation == ninfer::exl3::Exl3TargetProjectionOperator::o &&
                      timing.in_features == 6144 && timing.out_features == 5120) ||
                     (timing.operation == ninfer::exl3::Exl3TargetProjectionOperator::down &&
                      timing.in_features == 17408 && timing.out_features == 5120));
                const bool candidate_eligible =
                    gateup_candidate_eligible || down_candidate_eligible ||
                    o_k7_candidate_eligible || o_k6_candidate_eligible || down_k7_candidate_eligible ||
                    z_k6_candidate_eligible || qkv_k6_candidate_eligible ||
                    q_k6_candidate_eligible || kv_candidate_eligible ||
                    k5_batch_candidate_eligible;
                const bool candidate_reported =
                    timing.topology ==
                    Exl3TargetProjectionTopology::small_m_mma_split;
                if (candidate_reported) {
                    require(candidate_eligible && timing.rows == block_len &&
                                timing.calls == 1,
                            "target projection timing candidate topology mismatch layer=" +
                                std::to_string(timing.layer) + " op=" +
                                std::to_string(static_cast<int>(timing.operation)) +
                                " K=" + std::to_string(timing.K) + " rows=" +
                                std::to_string(timing.rows) + " calls=" +
                                std::to_string(timing.calls) + " phase=" +
                                std::to_string(static_cast<int>(timing.phase)));
                } else {
                    // An eligible group may retain this path when cooperative
                    // capacity is insufficient.  The external proof checker
                    // separately requires every enabled family on the
                    // qualification GPU.
                    require(timing.rows == (attempt ? block_len : 1) &&
                                timing.calls == (attempt ? block_len : 1) &&
                                timing.topology ==
                                    (attempt
                                         ? Exl3TargetProjectionTopology::m1_per_row
                                         : Exl3TargetProjectionTopology::m1),
                            "target projection timing M1 topology mismatch");
                }
            }
            out.target_projection_timings.insert(
                out.target_projection_timings.end(), timings.begin(), timings.end());
            out.target_projection_timing_resolution_us +=
                std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - resolution0).count();
        }
        out.reads.push_back(rec);
        if (r == 0) TtftTrace::mark("first_verified_round", spec->position(), out.spec_tokens.back());
        out.attempted_verification_rows +=
            static_cast<std::size_t>(rec.attempted_verification_rows);
        out.replay_rows += static_cast<std::size_t>(rec.replay_rows);
        out.retained_rows += static_cast<std::size_t>(rec.retained_rows);
        out.state_reconstruction_rows +=
            static_cast<std::size_t>(rec.state_reconstruction_rows);
        if (handoff_profile) {
            handoff_profile->end(full_round);
            handoff_profile->resolve();
            if (r == 0)
                handoff_profile->sample_memory(
                    "proposal_verify_complete", r, &target, &draft, spec.get(),
                    kTapStageBytes);
        }
    }
    out.resident_request_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - resident0).count();
    out.target_projection_raw_loop_wall_us =
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - loop0).count();
    out.spec_loop_us = out.target_projection_raw_loop_wall_us -
        out.target_projection_timing_resolution_us;
    out.emitted_tokens = out.spec_tokens.size();
    out.processed_tokens = static_cast<std::size_t>(abs_pos - static_cast<int>(prefix.size()));
    require(out.output_budget == 0 ||
                out.processed_tokens >= static_cast<std::size_t>(out.output_budget),
            "E5A4 maximum reads exhausted before output budget");
    out.useful_committed = out.output_budget == 0
        ? out.processed_tokens
        : static_cast<std::size_t>(out.output_budget);
    out.surplus_retained = out.processed_tokens - out.useful_committed;
    const std::size_t recorded_processes = std::accumulate(
        out.reads.begin(), out.reads.end(), std::size_t{0},
        [](std::size_t total, const ReadRecord& record) {
            return total + static_cast<std::size_t>(record.committed);
        });
    require(recorded_processes == out.processed_tokens,
            "pending protocol per-read decode accounting mismatch");
    require(out.processed_tokens + 1 == out.emitted_tokens,
            "pending protocol must retain exactly one emitted token");
    require(out.retained_rows == out.processed_tokens,
            "verifier retained-row accounting mismatch");
    if (out.transaction_route) {
        const std::size_t expected_attempted_rows = std::accumulate(
            out.reads.begin(), out.reads.end(), std::size_t{0},
            [](std::size_t total, const ReadRecord& record) {
                return total + static_cast<std::size_t>(record.attempted_block);
            });
        require(out.attempted_verification_rows == expected_attempted_rows,
                "transaction verifier attempted-row accounting mismatch");
    }
    if (out.retain_prefix_route) {
        require(out.replay_rows == 0,
                "retained-prefix verifier unexpectedly replayed model rows");
        const std::size_t expected_reconstruction_rows = std::accumulate(
            out.reads.begin(), out.reads.end(), std::size_t{0},
            [](std::size_t total, const ReadRecord& record) {
                return total + (record.rejpos < 0
                    ? 0u
                    : static_cast<std::size_t>(record.retained_rows));
            });
        require(out.state_reconstruction_rows == expected_reconstruction_rows,
                "retained-prefix verifier reconstruction-row accounting mismatch");
    }
    if (oscar) {
        spec->oscar_routing_counts(spec_oscar_full, spec_ordinary_full);
        spec_gdn = spec->oscar_telemetry().gdn_oscar_dispatches;
    }
    if (final_state_check)
        spec_final_state = e5a4_authoritative_state_snapshot(*spec);
    if (measure_teardown) {
        cuda_check(cudaDeviceSynchronize(),
                   "synchronize before measured spec context teardown");
        const auto teardown_begin = std::chrono::steady_clock::now();
        spec.reset();
        cuda_check(cudaDeviceSynchronize(),
                   "synchronize after measured spec context teardown");
        out.spec_context_teardown_us =
            std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - teardown_begin).count();
    }
    }  // end spec phase: context destroyed, long KV freed before control
    if (handoff_profile)
        handoff_profile->sample_memory(
            "spec_context_destroyed", -1, &target, &draft, nullptr, kTapStageBytes);
    // ---- Control phase (no tap capture): same prefix, same token count. ----
    {
        TtftTrace::scope("control_after_spec", static_cast<int>(prefix.size()));
        TtftTrace::mark("context_begin");
        const auto control_resident0 = std::chrono::steady_clock::now();
        auto control = target.create_context(final_state_check);
        TtftTrace::mark("context_complete", control->position(), -1, true);
        if (handoff_profile)
            handoff_profile->sample_memory(
                "control_context_created", -1, &target, &draft, control.get(),
                kTapStageBytes);
        TtftTrace::mark("oscar_attach_begin");
        if (oscar) {
            require(control->try_enable_oscar_from_environment(), "E5A4 OSCAR enable (control) failed");
        }
        TtftTrace::mark("oscar_attach_complete", control->position(), -1, true);
        if (handoff_profile)
            handoff_profile->sample_memory(
                "control_oscar_enabled", -1, &target, &draft, control.get(),
                kTapStageBytes);
        CommitSink nosink;
        HandoffProfile::Handle control_total;
        if (handoff_profile)
            control_total = handoff_profile->begin(
                -1, "control", "control_total");
        HandoffProfile::Handle control_prefill;
        if (handoff_profile)
            control_prefill = handoff_profile->begin(-1, "control", "prefill");
        out.control_prefill_us = ingest_prefix(*control, prefix, nosink);
        if (handoff_profile) {
            handoff_profile->end(control_prefill);
            handoff_profile->sample_memory(
                "control_prefix_complete", -1, &target, &draft, control.get(),
                kTapStageBytes);
        }
        out.control_tokens.reserve(out.spec_tokens.size());
        const auto control_seed0 = std::chrono::steady_clock::now();
        TtftTrace::mark("seed_begin", control->position());
        out.control_tokens.push_back(sample_target(*control));
        out.control_resident_ttft_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - control_resident0).count();
        TtftTrace::mark("first_token", control->position(), out.control_tokens.back());
        out.control_seed_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - control_seed0).count();
        require(out.control_tokens.front() == out.spec_tokens.front(),
                "pending control initial seed mismatch");
        const auto control0 = std::chrono::steady_clock::now();
        HandoffProfile::Handle control_decode;
        if (handoff_profile)
            control_decode = handoff_profile->begin(
                -1, "control", "decode_total");
        for (std::size_t read = 0; read < out.reads.size(); ++read) {
            for (int step = 0; step < out.reads[read].committed; ++step) {
                control->decode(out.control_tokens.back());
                out.control_tokens.push_back(sample_target(*control));
            }
        }
        out.control_resident_request_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - control_resident0).count();
        out.control_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - control0).count();
        if (handoff_profile) {
            handoff_profile->end(control_decode);
            handoff_profile->end(control_total);
            handoff_profile->resolve();
            handoff_profile->sample_memory(
                "control_complete", -1, &target, &draft, control.get(),
                kTapStageBytes);
        }
        require(control->position() ==
                    static_cast<int>(prefix.size() + out.control_tokens.size() - 1),
                "pending control final token was not left pending");
        if (final_state_check) {
            const auto control_final_state =
                e5a4_authoritative_state_snapshot(*control);
            out.final_state_exact = spec_final_state == control_final_state;
            require(out.final_state_exact,
                    "adaptive block final authoritative state mismatch");
        }
    }
    if (handoff_profile)
        handoff_profile->sample_memory(
            "control_context_destroyed", -1, &target, &draft, nullptr,
            kTapStageBytes);
    out.correctness = (out.spec_tokens == out.control_tokens) &&
        out.final_state_exact;
    out.oscar_full = spec_oscar_full;
    out.ordinary_full = spec_ordinary_full;
    out.gdn_oscar = spec_gdn;
    cuda_check(cudaMemGetInfo(&out.free_end, &out.total_mem), "E5A4 meminfo end");
    return out;
}


void write_accept(const std::string& family, int ctx, int block_len,
                  const ConfigResult& res, int warmup,
                  std::ostream& summary, std::ostream& perread, std::ostream& hist) {
    long long proposed = 0, accepted = 0, committed = 0;
    int zero = 0, full = 0;
    std::vector<double> acc_list, draft_list, verify_list;
    std::vector<long long> rejpos_count(static_cast<std::size_t>(block_len - 1) + 1, 0);
    for (std::size_t i = 0; i < res.reads.size(); ++i) {
        const ReadRecord& rec = res.reads[i];
        perread << family << "," << ctx << "," << block_len << "," << rec.read << ","
                << rec.accepted << "," << rec.committed << "," << rec.rejpos << ","
                << std::fixed << std::setprecision(1) << rec.draft_us << "," << rec.verify_us
                << std::defaultfloat << "," << rec.target_decodes << ","
                << rec.proposal_ids << ',' << std::fixed << std::setprecision(1)
                << rec.round_us << ','
                << rec.verification_route << ',' << rec.attempted_verification_rows << ','
                << rec.replay_rows << ',' << rec.retained_rows << ','
                << rec.attempted_verification_rows + rec.replay_rows << ','
                << rec.state_reconstruction_rows << ',' << rec.attempted_block << "\n";
        if (static_cast<int>(i) < warmup) continue;
        proposed += rec.attempted_block - 1;
        accepted += rec.accepted;
        committed += rec.committed;
        acc_list.push_back(static_cast<double>(rec.accepted));
        draft_list.push_back(rec.draft_us);
        verify_list.push_back(rec.verify_us);
        if (rec.accepted == 0) ++zero;
        if (rec.accepted == rec.attempted_block - 1) ++full;
        const std::size_t bin = rec.rejpos < 0 ? 0 : static_cast<std::size_t>(rec.rejpos) + 1;
        rejpos_count[bin]++;
    }
    const double n = static_cast<double>(acc_list.size());
    require(n > 0, "E5A4 no measured reads");
    std::sort(acc_list.begin(), acc_list.end());
    const double acc_rate = static_cast<double>(accepted) / static_cast<double>(proposed);
    const double per_read = static_cast<double>(committed) / n;
    // Surplus retained rows are fixed-run state diagnostics, never useful
    // output throughput. With no semantic budget the two counts coincide.
    const double spec_tps = static_cast<double>(res.useful_committed) / (res.spec_loop_us / 1e6);
    const double tgt_tps = static_cast<double>(res.useful_committed) / (res.control_us / 1e6);
    summary << family << "," << ctx << "," << block_len << "," << acc_list.size() << ","
            << proposed << "," << accepted << "," << std::fixed << std::setprecision(4) << acc_rate
            << "," << std::setprecision(3) << per_read
            << "," << acc_list[acc_list.size() / 2]
            << "," << std::setprecision(4) << static_cast<double>(full) / n
            << "," << static_cast<double>(zero) / n
            << "," << std::setprecision(1) << median(draft_list) << "," << median(verify_list)
            << "," << res.spec_loop_us / 1000.0 << "," << res.control_us / 1000.0
            << "," << std::setprecision(2) << spec_tps << "," << tgt_tps << "," << spec_tps / tgt_tps
            << "," << res.prefill_us / 1000.0
            << "," << (res.correctness ? "PASS" : "FAIL")
                << "," << res.oscar_full << "," << res.ordinary_full << "," << res.gdn_oscar
                << "," << res.free_end << "," << res.total_mem
                << ",pending_anchor_v1," << res.emitted_tokens << ',' << res.processed_tokens
                << ',' << res.initial_seed_us << ','
                << res.verification_route << ','
                << res.transaction_setup_us << ',' << res.transaction_bytes << ','
                << res.attempted_verification_rows << ',' << res.replay_rows << ','
                << res.retained_rows << ','
                << res.attempted_verification_rows + res.replay_rows << ','
                << (res.prefill_us + res.initial_seed_us + res.transaction_setup_us +
                    res.spec_loop_us) / 1000.0 << ','
                << res.state_reconstruction_rows << ','
                << res.control_prefill_us / 1000.0 << ','
                << res.control_seed_us << ','
                << (res.control_prefill_us + res.control_seed_us) / 1000.0
                << ',' << res.resident_ttft_ms << ',' << res.resident_request_ms
                << ',' << res.control_resident_ttft_ms << ',' << res.control_resident_request_ms
                << ',' << res.draft_prefill_submitted_rows << ',' << res.draft_prefill_encoded_rows
                << ',' << res.draft_prefill_skipped_rows
                << ',' << (res.draft_prefill_overlap ? 1 : 0)
                << ',' << res.draft_prefill_overlap_staging_bytes
                << ',' << res.draft_prefill_overlap_join_us
                << ',' << (res.draft_prefill_overlap_guards ? 1 : 0)
                << ',' << res.output_budget << ',' << res.useful_committed
                << ',' << res.surplus_retained << ','
                << (static_cast<double>(res.useful_committed) /
                    (res.spec_loop_us / 1.0e6)) << ','
                << res.spec_context_teardown_us
                << "\n" << std::defaultfloat;
    for (std::size_t b = 0; b < rejpos_count.size(); ++b) {
        hist << family << "," << ctx << "," << block_len << ","
             << (b == 0 ? "full" : std::to_string(b)) << "," << rejpos_count[b] << "\n";
    }
    std::cout << "E5A4 family=" << family << " ctx=" << ctx << " block=" << block_len
              << " acc_rate=" << std::fixed << std::setprecision(4) << acc_rate
              << " per_read=" << std::setprecision(3) << per_read
              << " spec_tps=" << std::setprecision(2) << spec_tps
              << " tgt_tps=" << tgt_tps << " speedup_seq=" << spec_tps / tgt_tps
              << " correctness=" << (res.correctness ? "PASS" : "FAIL") << std::defaultfloat << "\n";
}

void write_position_confidence(const std::string& family, int ctx,
                               const ConfigResult& res,
                               std::ostream& confidence_out) {
    for (const auto& rec : res.reads) {
        require(rec.position_confidence.size() >=
                    static_cast<std::size_t>(rec.attempted_block - 1),
                "T73 confidence output misses a verified proposed position");
        for (std::size_t p = 0; p < rec.position_confidence.size(); ++p) {
            const auto& confidence = rec.position_confidence[p];
            const bool attempted = static_cast<int>(p) < rec.attempted_block - 1;
            const bool known = attempted && (rec.rejpos < 0 ||
                static_cast<int>(p) <= rec.rejpos);
            const bool matched = known &&
                (rec.rejpos < 0 || static_cast<int>(p) < rec.rejpos);
            const double edge_scale = std::max(
                {std::abs(static_cast<double>(confidence.selected_edge_score)),
                 std::abs(static_cast<double>(confidence.runner_up_edge_score)),
                 1.0e-6});
            const double unary_scale = std::max(
                {std::abs(static_cast<double>(confidence.best_unary_score)),
                 std::abs(static_cast<double>(confidence.runner_up_unary_score)),
                 1.0e-6});
            confidence_out
                << family << ',' << ctx << ',' << rec.read << ',' << p << ','
                << std::setprecision(std::numeric_limits<float>::max_digits10)
                << confidence.selected_edge_score << ','
                << confidence.runner_up_edge_score << ','
                << confidence.edge_margin << ','
                << std::setprecision(std::numeric_limits<double>::max_digits10)
                << static_cast<double>(confidence.edge_margin) / edge_scale << ','
                << std::setprecision(std::numeric_limits<float>::max_digits10)
                << confidence.selected_unary_score << ','
                << confidence.best_unary_score << ','
                << confidence.runner_up_unary_score << ','
                << confidence.unary_margin << ','
                << std::setprecision(std::numeric_limits<double>::max_digits10)
                << static_cast<double>(confidence.unary_margin) / unary_scale << ','
                << confidence.selected_candidate_rank << ','
                << (known ? 1 : 0) << ',' << (matched ? 1 : 0) << ','
                << rec.rejpos << ',' << rec.attempted_block << ',' << rec.accepted
                << '\n';
        }
    }
}

void write_confidence_policy(const std::string& family, int ctx,
                             const ConfigResult& res,
                             std::ostream& policy_out) {
    for (const auto& rec : res.reads) {
        require(rec.confidence_policy_active,
                "T73 policy output requires active policy telemetry");
        policy_out << family << ',' << ctx << ',' << rec.read << ','
                   << (rec.confidence_policy_burn_in ? 1 : 0) << ','
                   << (rec.confidence_policy_fallback ? 1 : 0) << ','
                   << rec.attempted_block << ','
                   << std::setprecision(std::numeric_limits<double>::max_digits10)
                   << rec.confidence_policy_decision_us;
        for (const double expected : rec.confidence_expected_committed)
            policy_out << ',' << expected;
        for (const double efficiency : rec.confidence_efficiency)
            policy_out << ',' << efficiency;
        policy_out << ',' << rec.accepted << ',' << rec.committed << ','
                   << rec.rejpos << ',' << rec.attempted_verification_rows << ','
                   << rec.replay_rows << ',' << rec.retained_rows << ','
                   << rec.state_reconstruction_rows << ',' << rec.draft_us << ','
                   << rec.verify_us << ',' << rec.round_us << '\n';
    }
}

// Focused H6 small-M sweep on the REAL target head weights (Stage E).
// Every M-row output row is checked against a dedicated M=1 evaluation of
// the same input row. Reports dispatch path, device time, wall time,
// per-row cost, and correctness for each M in {1,2,3,4,6,7,8}.
void run_h6sweep(Exl3TextModel& target, const std::vector<std::int64_t>& prompt,
                 std::ostream& out) {
    auto ctx = target.create_context(false);
    {  // Chunked ingest (E4 pattern): at most 16 rows/forward.
        const int c0 = std::min<int>(static_cast<int>(prompt.size()), 16);
        ctx->prefill(std::span<const std::int64_t>(prompt.data(), static_cast<std::size_t>(c0)));
        for (int i = c0; i < static_cast<int>(prompt.size()); ++i) ctx->decode(prompt[static_cast<std::size_t>(i)]);
    }
    const auto& weights = ctx->target_lm_head_weights();
    const auto& metadata = ctx->target_lm_head_metadata();
    const std::array<int, 8> kMs = {1, 2, 3, 4, 5, 6, 7, 8};
    std::cout << "E5A4 h6 dims in=" << metadata.in_features << " out=" << metadata.out_features
              << " K=" << metadata.K << " mul1=" << metadata.mul1 << "\n";
    // Deterministic F16 input rows (sine pattern, RMS ~1).
    std::vector<std::uint16_t> host_in(static_cast<std::size_t>(8) * kHidden);
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < kHidden; ++c) {
            const float v = std::sin(static_cast<float>(r * kHidden + c) * 0.001f);
            std::uint32_t bits = 0;
            std::memcpy(&bits, &v, sizeof(bits));
            const std::uint32_t exp = (bits >> 23) & 0xFF;
            std::uint16_t h = 0;
            if (exp == 0) { h = static_cast<std::uint16_t>((bits >> 16) & 0x8000); }
            else if (exp == 0xFF) { h = static_cast<std::uint16_t>(((bits >> 16) & 0x8000) | 0x7BFF); }
            else {
                const int new_exp = static_cast<int>(exp) - 127 + 15;
                if (new_exp >= 31) h = static_cast<std::uint16_t>(((bits >> 16) & 0x8000) | 0x7BFF);
                else if (new_exp <= 0) h = static_cast<std::uint16_t>((bits >> 16) & 0x8000);
                else h = static_cast<std::uint16_t>(((bits >> 16) & 0x8000) | (new_exp << 10) | ((bits >> 13) & 0x3FF));
            }
            host_in[static_cast<std::size_t>(r) * kHidden + c] = h;
        }
    }
    DeviceBuffer dev_in(host_in.size() * sizeof(std::uint16_t));
    DeviceBuffer dev_out(static_cast<std::size_t>(8) * kVocab * sizeof(std::uint16_t));
    DeviceBuffer dev_ref(static_cast<std::size_t>(kVocab) * sizeof(std::uint16_t));
    cuda_check(cudaMemcpy(dev_in.get(), host_in.data(), host_in.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice), "E5A4 h6 input upload");
    Exl3CudaLinearWorkspace ws(kHidden, kVocab, 8);
    auto* in_ptr = static_cast<std::uint16_t*>(dev_in.get());
    auto* out_ptr = static_cast<std::uint16_t*>(dev_out.get());
    auto* ref_ptr = static_cast<std::uint16_t*>(dev_ref.get());
    bool all_rows_pass = true;
    for (const int m : kMs) {
        const char* dispatch = ws.dispatch_name(metadata, m);
        cuda_check(cudaDeviceSynchronize(), "E5A4 h6 pre sync");
        ws.forward(weights, metadata, in_ptr, out_ptr, m);
        cuda_check(cudaDeviceSynchronize(), "E5A4 h6 post sync");
        std::vector<std::uint16_t> got(static_cast<std::size_t>(m) * kVocab);
        cuda_check(cudaMemcpy(got.data(), dev_out.get(), got.size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost), "E5A4 h6 download");
        // Correctness: every row r of the M-row output vs a dedicated M=1
        // evaluation of the same input row.
        double row0_max = 0.0, row0_se = 0.0, row0_sn = 0.0;
        double all_max = 0.0, all_se = 0.0, all_sn = 0.0;
        std::vector<std::uint16_t> ref(kVocab, 0);
        for (int r = 0; r < m; ++r) {
            cuda_check(cudaDeviceSynchronize(), "E5A4 h6 ref pre sync");
            ws.forward(weights, metadata, in_ptr + static_cast<std::size_t>(r) * kHidden,
                       ref_ptr, 1);
            cuda_check(cudaDeviceSynchronize(), "E5A4 h6 ref post sync");
            cuda_check(cudaMemcpy(ref.data(), dev_ref.get(), ref.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost), "E5A4 h6 ref download");
            for (int i = 0; i < kVocab; ++i) {
                const double ga = static_cast<double>(half_to_float(got[static_cast<std::size_t>(r) * kVocab + i]));
                const double ba = static_cast<double>(half_to_float(ref[static_cast<std::size_t>(i)]));
                const double d = ga - ba;
                all_max = std::max(all_max, std::abs(d));
                all_se += d * d;
                all_sn += ba * ba;
                if (r == 0) {
                    row0_max = std::max(row0_max, std::abs(d));
                    row0_se += d * d;
                    row0_sn += ba * ba;
                }
            }
        }
        // Timed reps: device events (median) + wall clock around the loop.
        for (int warmup = 0; warmup < 5; ++warmup)
            ws.forward(weights, metadata, in_ptr, out_ptr, m);
        cuda_check(cudaDeviceSynchronize(), "E5A5H h6 warmup sync");
        cudaEvent_t a = nullptr, b = nullptr;
        cuda_check(cudaEventCreate(&a), "E5A4 h6 event a");
        cuda_check(cudaEventCreate(&b), "E5A4 h6 event b");
        std::vector<double> samples;
        const auto w0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 21; ++rep) {
            cuda_check(cudaEventRecord(a), "E5A4 h6 rec a");
            ws.forward(weights, metadata, in_ptr, out_ptr, m);
            cuda_check(cudaEventRecord(b), "E5A4 h6 rec b");
            cuda_check(cudaEventSynchronize(b), "E5A4 h6 sync b");
            float ms = 0.0f;
            cuda_check(cudaEventElapsedTime(&ms, a, b), "E5A4 h6 elapsed");
            samples.push_back(static_cast<double>(ms) * 1000.0);
        }
        const auto w1 = std::chrono::steady_clock::now();
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        const double med = median(samples);
        const double wall_us = std::chrono::duration<double, std::micro>(w1 - w0).count() / 21.0;
        const double row0_rel = row0_sn > 0 ? std::sqrt(row0_se / row0_sn) : 0.0;
        const double all_rel = all_sn > 0 ? std::sqrt(all_se / all_sn) : 0.0;
        all_rows_pass = all_rows_pass && std::isfinite(all_rel) &&
                        all_max <= 0.0015 && all_rel <= 0.0007;
        out << m << "," << dispatch << "," << metadata.K << "," << metadata.in_features << ","
            << metadata.out_features << "," << std::fixed << std::setprecision(1) << med << ","
            << wall_us << "," << med / m << "," << std::setprecision(12) << row0_max << ","
            << row0_rel << "," << all_max << "," << all_rel << "\n";
        out.flush();
        std::cout << "E5A4 h6 M=" << m << " dispatch=" << dispatch
                  << " med_us=" << std::fixed << std::setprecision(1) << med
                  << " wall_us=" << wall_us << " per_row_us=" << med / m
                  << " row0_maxabs=" << std::setprecision(12) << row0_max
                  << " row0_rel=" << row0_rel
                  << " allrow_maxabs=" << all_max
                  << " allrow_rel=" << all_rel
                  << std::defaultfloat << "\n";
    }
    require(all_rows_pass, "H6 sweep failed existing EXL3 M1 comparison tolerance");
}

void write_target_projection_timing(const std::string& family, int ctx,
                                    int block_len, const ConfigResult& result,
                                    std::ostream& out) {
    double gpu_projection_us = 0.0;
    for (const auto& record : result.target_projection_timings) {
        gpu_projection_us += record.microseconds;
        out << family << ',' << ctx << ',' << block_len << ',' << record.round << ','
            << target_projection_phase_name(record.phase) << ',' << record.layer << ','
            << target_projection_operator_name(record.operation) << ',' << record.rows << ','
            << record.K << ',' << record.in_features << ',' << record.out_features << ','
            << target_projection_topology_name(record.topology) << ',' << record.calls << ','
            << std::fixed << std::setprecision(3) << record.microseconds << '\n';
    }
    out.flush();
    std::cout << "TARGET_PROJECTION_TIMING PASS ctx=" << ctx
              << " B=" << block_len
              << " records=" << result.target_projection_timings.size()
              << " gpu_projection_us=" << std::fixed << std::setprecision(3)
              << gpu_projection_us
              << " loop_excluding_resolution_us=" << result.spec_loop_us
              << " raw_loop_wall_us=" << result.target_projection_raw_loop_wall_us
              << " control_us=" << result.control_us
              << " setup_us=" << result.target_projection_timing_setup_us
              << " resolution_us=" << result.target_projection_timing_resolution_us
              << std::defaultfloat << '\n';
}

#include "test_exl3_h6_qualification.h"
#include "test_exl3_target_k6_oracle.h"
#include "test_exl3_target_k6_qualification.h"
#include "test_exl3_target_m1_k6_qualification.h"
#include "test_exl3_target_down_k6_oracle.h"
#include "test_exl3_target_down_k6_qualification.h"
#include "test_exl3_target_o_k7_oracle.h"
#include "test_exl3_target_o_k7_qualification.h"
#include "test_exl3_target_down_k7_oracle.h"
#include "test_exl3_target_down_k7_qualification.h"
#include "test_exl3_k5_oracle.h"
#include "test_exl3_draft_small_m_qualification.h"
#include "test_exl3_target_transaction.h"
#include "test_exl3_fast_device_transaction.h"
#include "test_exl3_pending_qualification.h"
#include "test_exl3_target_continuation.h"
#include "test_exl3_context_isolation.h"
#include "test_exl3_current_graph_state.h"
#include "test_exl3_graph_plan_boundary.h"
#include "test_exl3_prefix_retention.h"
#include "test_exl3_branch_reference.h"
#include "test_exl3_exact_host_state.h"
#include "test_exl3_outer_reference.h"
#include "test_exl3_turboangle_host.h"
#include "test_exl3_turboangle_l1.h"
#include "test_exl3_hierarchical_sync.h"
#include "test_exl3_hierarchical_profile.h"
#include "test_exl3_hierarchical_async.h"
#include "test_exl3_hierarchical_adaptive.h"
#include "test_exl3_resident_root.h"
#include "test_exl3_resident_root_t3.h"
#include "test_exl3_host_kv_batch_sync.h"
#include "test_exl3_host_kv_batch_sync_t3.h"
#include "test_exl3_host_kv_batch_copy.h"
#include "test_exl3_host_kv_batch_copy_t3.h"
#include "test_exl3_host_kv_pinned_chunks.h"
#include "test_exl3_host_kv_pinned_d2h.h"
#include "test_exl3_eager_mlp_gateup_concurrency.h"
#include "test_exl3_small_m_fused_gate_up_transform.h"
#include "test_exl3_prefill_k8_kv_async_a.h"
#include "test_exl3_exact_attention_q_shared.h"
#include "test_exl3_exact_attention_q_shared_t3.h"
#include "test_exl3_exact_attention_k_half2.h"
#include "test_exl3_exact_attention_k_half2_t3.h"
#include "test_exl3_exact_attention_v_half2.h"
#include "test_exl3_exact_attention_v_half2_t3.h"
#include "test_exl3_exact_attention_gqa_pair.h"
#include "test_exl3_exact_attention_gqa_pair_t3.h"
#include "test_exl3_exact_attention_gqa_triple.h"
#include "test_exl3_exact_attention_gqa_six.h"
#include "test_exl3_exact_attention_gqa_six_scores_t3.h"
#include "test_exl3_exact_attention_gqa_triple_t3.h"
#include "test_exl3_exact_attention_gqa_triple_values128.h"
#include "test_exl3_exact_attention_gqa_triple_values128_t3.h"
#include "test_exl3_exact_attention_gqa_triple_values4.h"
#include "test_exl3_exact_attention_gqa_six_softmax_triple_values.h"
#include "test_exl3_exact_attention_gqa_six_softmax_triple_values_v_tile.h"
#include "test_exl3_exact_attention_gqa_six_packed_triples.h"
#include "test_exl3_exact_attention_gqa_six_extent_shards.h"
#include "test_exl3_exact_attention_gqa_six_score_k_tile64.h"
#include "test_exl3_exact_attention_gqa_triple_softmax_staged.h"
#include "test_exl3_exact_attention_gqa_triple_softmax_staged_t3.h"
#include "test_exl3_prefill_async_all.h"
#include "test_exl3_prefill_async_all_t3.h"
#include "test_exl3_prefill_direct_tiles64.h"
#include "test_exl3_prefill_persisting_l2.h"
#include "test_exl3_prefill_attention_chain_graph.h"
#include "test_exl3_prefill_attention_shared_scores.h"
#include "test_exl3_prefill_direct_tiles64_t3.h"
#include "test_exl3_prefill_reduce_shfl.h"
#include "test_exl3_prefill_reduce_shfl_t3.h"
#include "test_exl3_prefill_bundle.h"
#include "test_exl3_gdn_pair_columns.h"
#include "test_exl3_gdn_pair_columns_t3.h"
#include "test_exl3_gdn_pair_vector_io.h"
#include "test_exl3_gdn_dual_input_transform.h"
#include "test_exl3_gdn_quad_columns.h"
#include "test_exl3_oscar_fused_kv.h"
#include "test_exl3_oscar_fused_kv_t3.h"
#include "test_exl3_exact_continuation.h"
#include "test_exl3_outer_batched.h"
#include "test_exl3_exact_attention.h"
#include "test_exl3_numeric_attention_tiled_t71.h"
#include "test_exl3_prefix_retention_odd.h"
#include "test_exl3_transaction_retention.h"
#include "test_exl3_transaction_retention_odd.h"
#include "test_exl3_verification_width.h"
#include "test_exl3_prefill_gateup.h"
#include "test_exl3_prefill_k6_gateup_warpgroup.h"
#include "test_exl3_prefill_k6_gateup_n32_pair_cta.h"
#include "test_exl3_prefill_k6_fast_decode.h"
#include "test_exl3_prefill_k6_rowpair_n64.h"
#include "test_exl3_prefill_k6_down_rowpair.h"
#include "test_exl3_prefill_shape4_n64.h"
#include "test_exl3_prefill_reduce_min_barriers.h"
#include "test_exl3_prefill_k7_tiles64_exact_splits.h"
#include "test_exl3_initial16.h"
#include "test_exl3_prefill_chunk.h"
#include "test_exl3_target_z_k6_qualification.h"
#include "test_exl3_target_qkv_k6_qualification.h"
#include "test_exl3_target_q_k6_qualification.h"
#include "test_exl3_target_o_k6_qualification.h"
#include "test_exl3_target_gateup_k5_qualification.h"
#include "test_exl3_target_k5_small_m_batch.h"
#include "test_exl3_target_k6_m1_simt.h"
#include "test_exl3_target_kv_qualification.h"
#include "test_exl3_fresh_draft_prefill.h"
#include "test_exl3_wide_prefill.h"
#include "test_exl3_cross_request_projection_batch.h"
#include "test_exl3_packed_q_projection.h"
#include "test_exl3_target_owner_retirement.h"
#include "test_exl3_projection_reconstruct_gemm.h"
#include "test_exl3_projection_reconstruct_route.h"
#include "test_exl3_numeric_attention_route_t71b.h"
#include "test_exl3_first_divergence.h"
#include "test_exl3_reconstructed_exact.h"
#include "test_exl3_reconstructed_exact_k6_gate_up.h"
#include "test_exl3_reconstructed_exact_route.h"
#include "test_exl3_reconstructed_exact_authority.h"
#include "test_exl3_native16.h"
#include "test_exl3_native16_authority.h"
#include "exl3_host_residency.h"
#include "test_exl3_real_dflash_execution.h"
#include "test_exl3_draft_execution_ownership.h"
#include "test_exl3_k6_stream_reduction.h"
#include "test_exl3_extended_stream_reduction.h"
#include "test_exl3_device_prefix_ownership.h"
#include "test_exl3_draft_c2_workload.h"
#include "test_exl3_history_diagnostic.h"
#include "test_exl3_media_state.h"
#include "test_exl3_media_draft.h"
#include "test_exl3_recurrent_export_authority.h"
#include "test_exl3_real_dflash_screens.h"
#if defined(NINFER_V1_EXPORT) || defined(NINFER_V1_CONSUME)
#include "test_exl3_v1_oracle_state.h"
#else
#include "test_exl3_wide_prefill_state.h"
#endif
#include "test_exl3_vericache_request.h"
#include "test_exl3_exact_paged.h"
#include "test_exl3_native64k_prefix_switch.h"
#include "test_exl3_host_stream.h"
#include "test_exl3_compact_l0.h"
#include "test_exl3_vericache_queue.h"
#include "test_exl3_exact_wide.h"
#include "test_exl3_wide_request.h"
#include "test_exl3_outer_stop.h"
#include "test_exl3_host_residency.h"
#include "test_exl3_token_replay.h"
#include "test_exl3_capacity.h"
#include "test_exl3_draft_host_ring.h"
#include "test_exl3_prefix_index.h"
#include "test_exl3_prefix_serving_t72.h"
#include "test_exl3_prefix_serving_t72_t1.h"
#include "test_exl3_context_reuse.h"
#include "test_exl3_compact_request.h"
#include "test_exl3_compact_perf.h"
#include "test_exl3_l2_window_experiment.h"
#include "test_exl3_hierarchical_l2_windows_t70.h"
#include "test_exl3_hierarchical_l2_windows_t70_t1.h"
#include "test_exl3_quality_probe.h"

}  // namespace

int main() {
    const auto entry_stamp = TtftTrace::stamp();
    try {
        const auto target_path = env("NINFER_EXL3_TARGET_PATH");
        const auto draft_path = env("NINFER_EXL3_DFLASH2_PATH");
        const std::string mode = env("NINFER_E5A4_MODE");
#if defined(NINFER_V1_EXPORT) || defined(NINFER_V1_CONSUME)
        require(mode == "prefillwidestate", "V1 executable only supports exact state qualification");
#endif
        if (mode == "graphplanboundary") {
            const std::string out_path = env("NINFER_E5A4_OUT");
            require(!out_path.empty(), "graphplanboundary needs OUT");
            run_graph_plan_boundary(out_path);
            return 0;
        }
        if (mode == "oscardefaultconstruction") {
            run_oscar_default_construction(env("NINFER_E5A4_OUT"));
            return 0;
        }
        int targetrequest_count = 0, targetrequest_outputs = 0;
        int targetrequest_context_count = 0;
        std::size_t targetrequest_source_offset = 0;
        bool targetrequest_context_mode = false;
        bool targetrequest_ordinary_fp16 = false;
        bool targetrequest_exact_host_kv = false;
        std::vector<std::int64_t> targetrequest_prefix;
        std::string targetrequest_summary_path, targetrequest_tokens_path,
            targetrequest_rounds_path;
        if (mode == "targetrequest") {
            const bool decode_projection_diagnostic=
                !env("NINFER_E5A4_TARGETREQUEST_DECODE_PROJECTION_PROFILE").empty();
            for (const auto* name : {"NINFER_E5A4_TTFT_OUT", "NINFER_E5A4_HANDOFF_PROFILE_PREFIX",
                 "NINFER_EXL3_TARGET_PROJECTION_TIMING_OUT", "NINFER_E5A4_RING_CHECK_OUT",
                 "NINFER_E5A5H_CAPTURE"})
                require(env(name).empty(), "targetrequest forbids profiling outputs");
            require(decode_projection_diagnostic?
                    env("NINFER_EXL3_TARGET_PROJECTION_TIMING")=="1":
                    (env("NINFER_EXL3_TARGET_PROJECTION_TIMING").empty() ||
                     env("NINFER_EXL3_TARGET_PROJECTION_TIMING")=="0"),
                "targetrequest projection timing requires its explicit diagnostic output");
            for (const auto* name : {"NINFER_E5A4_GRAPH",
                 "NINFER_EXL3_GRAPH", "NINFER_E5A4_NVTX"})
                require(env(name).empty() || env(name) == "0", "targetrequest eager uninstrumented only");
            const auto ordinary_fp16=env("NINFER_TARGETREQUEST_ORDINARY_FP16");
            require(ordinary_fp16.empty() || ordinary_fp16=="0" || ordinary_fp16=="1",
                    "targetrequest ordinary FP16 option must be 0 or 1");
            targetrequest_ordinary_fp16=ordinary_fp16=="1";
            targetrequest_exact_host_kv=
                !targetrequest_ordinary_fp16 &&
                env("NINFER_EXL3_EXACT_HOST_KV")=="1";
            require(env("NINFER_EXL3_TARGET_QKV_K6_SMALL_M") == "1" &&
                    env("NINFER_DFLASH2_PREFILL_WINDOW") == "0",
                    "targetrequest requires P9 QKV1/P10 off");
            if(targetrequest_ordinary_fp16) {
                require((env("NINFER_OSCAR_EXL3").empty() || env("NINFER_OSCAR_EXL3")=="0") &&
                        (env("NINFER_EXL3_EXACT_HOST_KV").empty() || env("NINFER_EXL3_EXACT_HOST_KV")=="0"),
                        "targetrequest ordinary FP16 requires device KV and OSCAR off");
            } else if(targetrequest_exact_host_kv) {
                require(env("NINFER_OSCAR_EXL3").empty() || env("NINFER_OSCAR_EXL3")=="0",
                        "targetrequest exact HostKV requires OSCAR off");
            } else require(env("NINFER_OSCAR_EXL3") == "1",
                    "targetrequest requires exact HostKV, canonical OSCAR, or ordinary FP16 device KV");
            const auto* allow_projection_nvtx =
                std::getenv("NINFER_EXL3_TARGETREQUEST_ALLOW_PROJECTION_NVTX");
            require(std::getenv("NINFER_EXL3_PROJECTION_NVTX") == nullptr ||
                    (allow_projection_nvtx &&
                     std::strcmp(allow_projection_nvtx, "1") == 0),
                    "targetrequest forbids projection NVTX unless explicitly diagnostic-gated");
            const std::string context_count_setting = env("NINFER_TARGETREQUEST_CONTEXT_COUNT");
            targetrequest_context_mode = !context_count_setting.empty();
            require(!targetrequest_ordinary_fp16 || !targetrequest_context_mode,
                    "targetrequest ordinary FP16 supports physical C1 only");
            require(targetrequest_context_mode || env("NINFER_TARGETREQUEST_SOURCE_OFFSET").empty(),
                    "targetrequest source offset requires explicit context count");
            if (targetrequest_context_mode) {
                require(context_count_setting == "1" || context_count_setting == "2" ||
                            context_count_setting == "4" || context_count_setting == "8",
                        "targetrequest context count must be exactly 1, 2, 4, or 8");
                targetrequest_context_count = std::stoi(context_count_setting);
                require(env("NINFER_E5A4_MAXCTX") == "1024",
                        "targetrequest context benchmark requires maxctx1024");
                require(env("NINFER_EXL3_PREFILL_ROWPAIR_K6") == "0",
                        "targetrequest context benchmark requires rowpair0");
                const auto prefill_chunk = env("NINFER_EXL3_PREFILL_CHUNK");
                if (prefill_chunk == "128") {
                    const auto slab = env("NINFER_EXL3_GDN_WIDE_SLAB");
                    require(env("NINFER_EXL3_WIDE_PREFILL") == "1" &&
                                env("NINFER_EXL3_PREFILL_WIDE64") == "1" &&
                                env("NINFER_EXL3_PREFILL_WIDE128") == "1" &&
                                env("NINFER_EXL3_PREFILL_WIDE256") == "0" &&
                                env("NINFER_EXL3_PREFILL_WIDE512") == "0" &&
                                env("NINFER_EXL3_PREFILL_WIDE1024") == "0" &&
                                env("NINFER_EXL3_PREFILL_STAGED_REDUCTION") == "1" &&
                                env("NINFER_EXL3_PREFILL_STAGED_SHAPE4") == "1" &&
                                env("NINFER_EXL3_PREFILL_GDN_RESIDENT") == "1" &&
                                env("NINFER_EXL3_SHARED_ACCUM") == "1" &&
                                env("NINFER_EXL3_SHARED_TRANSFORM") == "1" &&
                                env("NINFER_EXL3_SHARED_LAYER_SCRATCH") == "1" &&
                                (slab.empty() || slab == "1"),
                            "targetrequest context benchmark requires the P17 wide128 overlay");
                } else {
                    require(prefill_chunk == "1024" && targetrequest_context_count == 1,
                            "targetrequest context benchmark allows chunk1024 only for count1");
                }
            } else {
                require(env("NINFER_EXL3_PREFILL_CHUNK") == "1024",
                        "targetrequest qualified chunk1024 only");
            }
            // Explicit output count includes the seed; the final sampled token remains pending.
            const auto strict_count = [](const char* name) {
                const auto text = env(name);
                require(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos,
                        std::string("targetrequest invalid integer: ") + name);
                const auto value = std::stoll(text);
                require(value >= 1 && value <= std::numeric_limits<int>::max(), "targetrequest integer extent");
                return static_cast<int>(value);
            };
            targetrequest_count = strict_count("NINFER_E5A4_CONTEXTS");
            targetrequest_outputs = strict_count("NINFER_TARGETREQUEST_OUTPUTS");
            targetrequest_prefix = load_ids(env("NINFER_E5A4_PROMPT_FILE"));
            if (targetrequest_context_mode) {
                const auto offset_text = env("NINFER_TARGETREQUEST_SOURCE_OFFSET");
                require(offset_text.empty() ||
                            offset_text.find_first_not_of("0123456789") == std::string::npos,
                        "targetrequest source offset must be a nonnegative integer");
                targetrequest_source_offset = offset_text.empty() ? 0 : std::stoull(offset_text);
                require(targetrequest_context_count == 1 || targetrequest_source_offset == 0,
                        "targetrequest multipath source offset must be zero");
                require(targetrequest_outputs == 32,
                        "targetrequest context benchmark requires 32 outputs including seed");
                require(targetrequest_count >= 32 &&
                            static_cast<std::size_t>(targetrequest_count) <= targetrequest_prefix.size(),
                        "targetrequest context benchmark prefix extent");
                const std::size_t stride = 512;
                require(targetrequest_source_offset <= targetrequest_prefix.size() &&
                            static_cast<std::size_t>(targetrequest_context_count - 1) <=
                                (targetrequest_prefix.size() - targetrequest_source_offset) / stride &&
                            static_cast<std::size_t>(targetrequest_count) <=
                                targetrequest_prefix.size() - targetrequest_source_offset -
                                    static_cast<std::size_t>(targetrequest_context_count - 1) * stride,
                        "targetrequest context benchmark source slices");
                for (int stream = 0; stream < targetrequest_context_count; ++stream) {
                    const std::size_t begin = targetrequest_source_offset +
                        static_cast<std::size_t>(stream) * stride;
                    for (int i = 0; i < targetrequest_count; ++i) {
                        const auto id = targetrequest_prefix[begin + static_cast<std::size_t>(i)];
                        require(id >= 0 && id < kVocab,
                                "targetrequest context benchmark token extent");
                        require(id != kMaskToken,
                                "targetrequest context benchmark source slice contains mask");
                    }
                    require(targetrequest_prefix[begin + static_cast<std::size_t>(targetrequest_count - 1)] != kMaskToken,
                            "targetrequest context benchmark prefix cannot end in mask");
                }
                require(static_cast<std::int64_t>(targetrequest_count) + targetrequest_outputs - 1 <= 1024,
                        "targetrequest context benchmark output extent");
            } else {
                require(targetrequest_count >= 32 && static_cast<std::size_t>(targetrequest_count) <= targetrequest_prefix.size() &&
                        static_cast<std::int64_t>(targetrequest_count) + targetrequest_outputs - 1 <= strict_count("NINFER_E5A4_MAXCTX"),
                        "targetrequest prefix/output extent");
                targetrequest_prefix.resize(targetrequest_count);
                for (const auto id : targetrequest_prefix) require(id >= 0 && id < kVocab, "targetrequest prefix token extent");
                require(targetrequest_prefix.back() != kMaskToken, "targetrequest prefix cannot end in mask");
            }
            targetrequest_summary_path = env("NINFER_E5A4_OUT");
            targetrequest_tokens_path = env("NINFER_TARGETREQUEST_TOKENS_OUT");
            require(!targetrequest_summary_path.empty() && !targetrequest_tokens_path.empty() && targetrequest_summary_path != targetrequest_tokens_path &&
                    !std::filesystem::exists(targetrequest_summary_path) && !std::filesystem::exists(targetrequest_tokens_path),
                    "targetrequest outputs must be separate new files");
            if (targetrequest_context_mode) {
                targetrequest_rounds_path = targetrequest_summary_path + ".rounds.csv";
                require(targetrequest_rounds_path != targetrequest_tokens_path &&
                            !std::filesystem::exists(targetrequest_rounds_path),
                        "targetrequest context benchmark round output must be new");
            }
        }
        std::unique_ptr<TtftTrace> ttft;
        if (!env("NINFER_E5A4_TTFT_OUT").empty()) {
            require(mode == "accept" || mode == "ttfttarget", "TTFT requires accept or ttfttarget mode");
            require(env("NINFER_E5A4_CONTEXTS").find(',') == std::string::npos,
                    "TTFT requires one context per process");
            ttft = std::make_unique<TtftTrace>(env("NINFER_E5A4_TTFT_OUT"), entry_stamp);
        }
        if (mode == "topkqual") {
            run_topk_unit_qualification(env("NINFER_E5A4_OUT"));
            return 0;
        }
        if (mode == "ringqual") {
            run_ring_attention_qualification(env("NINFER_E5A4_OUT"));
            return 0;
        }
        if (mode == "dflashdensekmajorqual") {
            run_dflash2_dense_kmajor_qualification();
            return 0;
        }
        if (mode == "exactattentionops") {
            TtftTrace::mark("cuda_begin");
            cuda_check(cudaSetDevice(0), "exactattentionops set device");
            cuda_check(cudaDeviceSynchronize(), "exactattentionops warmup sync");
            TtftTrace::mark("cuda_complete");
            run_exact_attention_operator_oracle();
            return 0;
        }
        if (mode == "exactattngqasixsoftmaxtriplevtileoperator") {
            TtftTrace::mark("cuda_begin");
            cuda_check(cudaSetDevice(0), "V-tile operator set device");
            cuda_check(cudaDeviceSynchronize(), "V-tile operator warmup sync");
            TtftTrace::mark("cuda_complete");
            run_exact_attention_gqa_six_softmax_triple_values_v_tile_operator();
            return 0;
        }
        if (mode == "gqasixsegmentedprefixop") {
            TtftTrace::mark("cuda_begin");
            cuda_check(cudaSetDevice(0), "segmented six-softmax operator set device");
            cuda_check(cudaDeviceSynchronize(),
                "segmented six-softmax operator warmup sync");
            TtftTrace::mark("cuda_complete");
            run_exact_attention_gqa_six_segmented_prefix_operator();
            return 0;
        }
        if (mode == "gqasixscorektile64op") {
            TtftTrace::mark("cuda_begin");
            cuda_check(cudaSetDevice(0), "GQA six-score K-tile64 operator set device");
            cuda_check(cudaDeviceSynchronize(),
                "GQA six-score K-tile64 operator warmup sync");
            TtftTrace::mark("cuda_complete");
            run_exact_attention_gqa_six_score_k_tile64_operator();
            return 0;
        }
        if (target_path.empty() ||
             (draft_path.empty() && mode != "targetownerretirement" && mode != "packedqprojection" && mode != "hostkvbatchmetadata" && mode != "targettxn" && mode != "targetcontinue" &&
             mode != "prefixretain" && mode != "prefixretainodd" && mode != "verifywidth" && mode != "ttfttarget" && mode != "targetrequest" && mode != "qualitylogits" && mode != "layermajorfailure" && mode != "k5prefilloperator" && mode != "fastdevicetxn" && mode != "hostkvgdngraphoracle" && mode != "hostkvmlptailgraphoracle" && mode != "hostkvrecurrenttraceoracle" && mode != "t79targetc2" && mode != "t80fp16serving" && mode != "t82fp16c2" && mode != "t83fp16c8" && mode != "t84servingcoord" && mode != "t85fp16q4" && mode != "t86coordc8" && mode != "t87batchadmit" && mode != "t88unequaltails" && mode != "t89residentprofile" && mode != "t90allocationchurn" && mode != "t92turnover" && mode != "t95atomicturnover" && mode != "t97concurrentprep" && mode != "t98pipelinedturnover" && mode != "t101multiturn" && mode != "contextisolation" && mode != "graphcurrentstate" && mode != "graphcurrentreplay" && mode != "prefillgateup" && mode != "prefillk6gateupwarpgroup" && mode != "prefillk6gateupn32paircta" && mode != "prefillk7tiles64exact" && mode != "initial16ops" && mode != "projectionorderedc2screen" && mode != "projectionroutematrixscreen" && mode != "gdnstageoracle" && mode != "targetgraphc2screen" && mode != "targetgraphc2lifecycle" && mode != "targetgraphresetreuse" && mode != "prefixindex" && mode != "prefixservingt72" && mode != "prefixservingt72t1" && mode != "t78cachedc2" && mode != "contextreuse" && mode != "contextreuset1")) {
            std::cerr << "E5A4 skipped: set NINFER_EXL3_TARGET_PATH"
                      << ((mode == "targettxn" || mode == "targetcontinue" ||
                           mode == "prefixretain" || mode == "prefixretainodd")
                              ? "\n" : " and NINFER_EXL3_DFLASH2_PATH\n");
            return 77;
        }
        const bool oscar_requested = env("NINFER_OSCAR_EXL3") == "1";
        TtftTrace::mark("cuda_begin");
        cuda_check(cudaSetDevice(0), "E5A4 set device");
        cuda_check(cudaDeviceSynchronize(), "E5A4 warmup sync");
        TtftTrace::mark("cuda_complete");
        std::unique_ptr<HandoffProfile> handoff_profile;
        const std::string handoff_profile_prefix =
            env("NINFER_E5A4_HANDOFF_PROFILE_PREFIX");
        if (!handoff_profile_prefix.empty()) {
            require(mode.empty() || mode == "accept",
                    "handoff profiling is available only in accept mode");
            require(env("NINFER_E5A4_VERIFIER") == "transaction_retain",
                    "handoff profiling requires transaction_retain verifier");
            handoff_profile = std::make_unique<HandoffProfile>(handoff_profile_prefix);
            handoff_profile->sample_memory(
                "cuda_ready", -1, nullptr, nullptr, nullptr);
        }
        const int max_ctx = env_int("NINFER_E5A4_MAXCTX", 4096);
        TtftTrace::mark("target_load_begin");
        ninfer::exl3::Exl3LoadOptions load_options;
        load_options.disable_dual_artifact_loading = env_int("NINFER_E5A4_DISABLE_DUAL_LOAD", 0) != 0;
        auto target = Exl3TextModel::load(target_path, max_ctx, load_options);
        TtftTrace::mark("target_load_complete", -1, -1, true);
        // Diagnostic only: keep the pinned draft resident while the target-only
        // GDN graph oracle executes.  This distinguishes model-residency/resource
        // composition from the later DFlash lane/controller integration without
        // changing either oracle path.
        std::shared_ptr<Exl3Dflash2DraftModel> gdn_graph_oracle_resident_draft;
        if (mode == "hostkvgdngraphoracle" &&
            env("NINFER_E5A4_GDN_GRAPH_WITH_DRAFT") == "1") {
            require(!draft_path.empty(),
                "HostKV GDN graph resident-draft oracle requires DFLASH2_PATH");
            gdn_graph_oracle_resident_draft = Exl3Dflash2DraftModel::load(draft_path);
        }
        const auto& load_stats = target->load_stats();
        std::cout << "EXL3_LOAD parallel_staged=" << load_stats.parallel_staged
                  << " c_bytes=" << load_stats.source_bytes[0]
                  << " d_bytes=" << load_stats.source_bytes[1]
                  << " audited_tensors=" << load_stats.audited_tensors
                  << " audited_bytes=" << load_stats.audited_bytes
                  << " prepare_ms=" << load_stats.staging_prepare_ms
                  << " release_ms=" << load_stats.staging_release_ms
                  << " payload_read_ms=" << load_stats.payload_read_ms
                  << " upload_ms=" << load_stats.upload_ms
                  << " staged_transfer_ms=" << load_stats.staged_transfer_ms
                  << " staging_host_bytes=" << load_stats.staging_host_bytes << '\n';
        if (mode == "layermajorfailure") {
            require(!oscar_requested &&
                    env("NINFER_EXL3_TEST_LAYER_MAJOR_FAIL_AFTER_LAYER") == "2" &&
                    env("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL") == "1",
                    "layer-major failure gate requires ordinary Fast90 and layer2 fault");
            const auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));
            require(ids.size()>=2065,"layer-major failure fixture extent");
            auto context=target->create_context(true);
            context->prefill(std::span<const std::int64_t>(ids.data(),16));
            bool fault_seen=false,decode_rejected=false;
            try {
                context->append_prefill_layer_major(
                    std::span<const std::int64_t>(ids.data()+16,2048));
            } catch (const std::exception& error) {
                fault_seen=std::string(error.what()).find(
                    "injected layer-major partial-layer failure")!=std::string::npos;
            }
            try { context->decode(ids[2064]); }
            catch (const std::exception& error) {
                decode_rejected=std::string(error.what()).find(
                    "failed layer-major prefill requires context reset")!=std::string::npos;
            }
            require(fault_seen && decode_rejected && context->position()==16,
                    "partial layer-major failure escaped or changed public position");
            require(_putenv_s("NINFER_EXL3_TEST_LAYER_MAJOR_FAIL_AFTER_LAYER","")==0,
                    "clear one-shot layer-major test fault");
            context->reset();
            context->prefill(std::span<const std::int64_t>(ids.data(),16));
            context->append_prefill_layer_major(
                std::span<const std::int64_t>(ids.data()+16,2048));
            require(context->position()==2064,
                    "reset did not restore layer-major prefill admission");
            context->decode(ids[2064]);
            require(context->position()==2065,
                    "post-reset layer-major decode position");
            cuda_check(cudaDeviceSynchronize(),
                       "complete layer-major failure/reset gate");
            std::cout << "LAYER_MAJOR_FAILURE PASS\n";
            return 0;
        }
        if (mode == "t79targetc2") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_T79_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "t79targetc2 needs code/prose prompts and output");
            require(oscar_requested, "t79targetc2 requires canonical OSCAR");
            run_t79_target_only_c2(*target, load_ids(code_file),
                load_ids(prose_file), output);
            return 0;
        }
        if (mode == "t80fp16serving") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_T80_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "t80fp16serving needs code/prose prompts and output");
            require(!oscar_requested,
                    "t80fp16serving is ordinary FP16, not OSCAR");
            run_t80_fp16_serving(*target, load_ids(code_file),
                load_ids(prose_file), output);
            return 0;
        }
        if (mode == "t82fp16c2") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_T82_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "t82fp16c2 needs code/prose prompts and output");
            require(!oscar_requested,
                    "t82fp16c2 is ordinary FP16, not OSCAR");
            run_t82_fp16_c2_serving(*target, load_ids(code_file),
                load_ids(prose_file), output);
            return 0;
        }
        if (mode == "t83fp16c8") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_T83_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "t83fp16c8 needs code/prose prompts and output");
            require(!oscar_requested,
                    "t83fp16c8 is ordinary FP16, not OSCAR");
            run_t83_fp16_c8_serving(*target, load_ids(code_file),
                load_ids(prose_file), output);
            return 0;
        }
        if (mode == "t85fp16q4") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_T85_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "t85fp16q4 needs code/prose prompts and output");
            require(!oscar_requested,
                    "t85fp16q4 is ordinary FP16, not OSCAR");
            run_t85_fp16_c8_serving(*target, load_ids(code_file),
                load_ids(prose_file), output);
            return 0;
        }
        if (mode == "t86coordc8") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_T86_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "t86coordc8 needs code/prose prompts and output");
            require(!oscar_requested,
                    "t86coordc8 is ordinary FP16, not OSCAR");
            run_t86_fp16_c8_serving(*target, load_ids(code_file),
                load_ids(prose_file), output);
            return 0;
        }
        if (mode == "t87batchadmit") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_T87_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "t87batchadmit needs code/prose prompts and output");
            require(!oscar_requested,
                    "t87batchadmit is ordinary FP16, not OSCAR");
            run_t87_batch_admission(*target, load_ids(code_file),
                load_ids(prose_file), output);
            return 0;
        }
        if (mode == "t89residentprofile") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_T89_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "t89residentprofile needs code/prose prompts and output");
            require(!oscar_requested,
                    "t89residentprofile is ordinary FP16, not OSCAR");
            run_t89_resident_profile(*target, load_ids(code_file),
                load_ids(prose_file), output);
            return 0;
        }
        if (mode == "t90allocationchurn") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto output = env("NINFER_T90_OUT");
            require(!code_file.empty() && !output.empty(),
                    "t90allocationchurn needs code prompt and output");
            require(!oscar_requested,
                    "t90allocationchurn is ordinary FP16, not OSCAR");
            run_t90_allocation_churn(*target, load_ids(code_file), output);
            return 0;
        }
        if (mode == "t92turnover") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_T92_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "t92turnover needs code/prose prompts and output");
            require(!oscar_requested,
                    "t92turnover is ordinary FP16, not OSCAR");
            run_t92_persistent_turnover(
                *target, load_ids(code_file), load_ids(prose_file), output);
            return 0;
        }
        if (mode == "t95atomicturnover") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_T95_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "t95atomicturnover needs code/prose prompts and output");
            require(!oscar_requested,
                    "t95atomicturnover is ordinary FP16, not OSCAR");
            run_t95_atomic_turnover(
                *target, load_ids(code_file), load_ids(prose_file), output);
            return 0;
        }
        if (mode == "t97concurrentprep") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_T97_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "t97concurrentprep needs code/prose prompts and output");
            require(!oscar_requested,
                    "t97concurrentprep is ordinary FP16, not OSCAR");
            run_t97_concurrent_prefix_prepare(
                *target, load_ids(code_file), load_ids(prose_file), output);
            return 0;
        }
        if (mode == "t98pipelinedturnover") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_T98_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "t98pipelinedturnover needs code/prose prompts and output");
            require(!oscar_requested,
                    "t98pipelinedturnover is ordinary FP16, not OSCAR");
            run_t98_pipelined_turnover(
                *target, load_ids(code_file), load_ids(prose_file), output);
            return 0;
        }
        if (mode == "t101multiturn") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_T101_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "t101multiturn needs code/prose prompts and output");
            require(!oscar_requested,
                    "t101multiturn is ordinary FP16, not OSCAR");
            run_t101_multiturn_prefix(
                *target, load_ids(code_file), load_ids(prose_file), output);
            return 0;
        }
        if (mode == "t88unequaltails") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_T88_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "t88unequaltails needs code/prose prompts and output");
            require(!oscar_requested,
                    "t88unequaltails is ordinary FP16, not OSCAR");
            run_t88_unequal_tail_fairness(
                *target, load_ids(code_file), load_ids(prose_file), output);
            return 0;
        }
        if (mode == "t84servingcoord") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_T84_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "t84servingcoord needs code/prose prompts and output");
            require(!oscar_requested,
                    "t84servingcoord is ordinary FP16, not OSCAR");
            run_t84_serving_coordinator(*target, load_ids(code_file),
                load_ids(prose_file), output);
            return 0;
        }
        if (mode == "qualitylogits") {
            const bool coherent_o_k7=env("NINFER_EXL3_COHERENT_O_K7")=="1";
            const bool coherent_down_k7=env("NINFER_EXL3_COHERENT_DOWN_K7")=="1";
            const bool coherent_wide_k6=
                env("NINFER_EXL3_TARGET_COHERENT_WIDE_K6")=="1";
            std::uint64_t coherent_wide_calls_before[5]{},
                coherent_wide_rows_before[5]{};
            for(int operation=0;operation<5;++operation) {
                coherent_wide_calls_before[operation]=
                    Exl3CudaLinearWorkspace::coherent_wide_k6_calls_for_test(operation);
                coherent_wide_rows_before[operation]=
                    Exl3CudaLinearWorkspace::coherent_wide_k6_rows_for_test(operation);
            }
            const auto coherent_down_k7_calls_before=
                Exl3CudaLinearWorkspace::process_coherent_down_k7_calls_for_test();
            const auto coherent_down_k7_rows_before=
                Exl3CudaLinearWorkspace::process_coherent_down_k7_rows_for_test();
            const auto coherent_o_calls_before=
                Exl3CudaLinearWorkspace::process_coherent_o_k7_calls_for_test();
            const auto coherent_o_rows_before=
                Exl3CudaLinearWorkspace::process_coherent_o_k7_rows_for_test();
            const auto input_path=env("NINFER_QUALITY_INPUT");
            const auto csv_path=env("NINFER_QUALITY_OUTPUT");
            const auto binary_path=env("NINFER_QUALITY_LOGITS_OUTPUT");
            require(!input_path.empty() && !csv_path.empty() && !binary_path.empty() &&
                    !std::filesystem::exists(csv_path) && !std::filesystem::exists(binary_path),
                    "qualitylogits requires new separate output files");
            const auto ids=load_ids(input_path);
            const int prefix_rows=env_int("NINFER_QUALITY_PREFIX",3072);
            const int scored_rows=env_int("NINFER_QUALITY_LABELS",128);
            require(prefix_rows>=32 && scored_rows>=1 &&
                    static_cast<std::size_t>(prefix_rows+scored_rows)<=ids.size() &&
                    prefix_rows+scored_rows<=max_ctx,"qualitylogits fixed common-history extent");
            for(int i=0;i<prefix_rows+scored_rows;++i)
                require(ids[static_cast<std::size_t>(i)]>=0 &&
                        ids[static_cast<std::size_t>(i)]<kVocab,
                        "qualitylogits vocabulary extent");
            auto context=target->create_context(true);
            std::vector<std::int64_t> prefix(ids.begin(),ids.begin()+prefix_rows);
            ingest_prefix(*context,prefix,CommitSink{});
            std::ofstream csv(csv_path),binary(binary_path,std::ios::binary);
            csv.imbue(std::locale::classic());
            csv << "ordinal,label,nll,top1,top2_margin,label_logit,logsumexp\n";
            double total_nll=0.0;
            for(int row=0;row<scored_rows;++row) {
                const auto logits=context->logits_host();
                require(logits.size()==kVocab,"qualitylogits vector extent");
                binary.write(reinterpret_cast<const char*>(logits.data()),
                             static_cast<std::streamsize>(logits.size()*sizeof(float)));
                const int label=static_cast<int>(ids[static_cast<std::size_t>(prefix_rows+row)]);
                double maximum=-std::numeric_limits<double>::infinity();
                double top1=-std::numeric_limits<double>::infinity();
                double top2=-std::numeric_limits<double>::infinity();
                int best=-1;
                for(std::size_t j=0;j<logits.size();++j) {
                    const double value=logits[j];
                    require(std::isfinite(value) || value==-std::numeric_limits<float>::infinity(),
                            "qualitylogits nonfinite unmasked logit");
                    if(value>maximum)maximum=value;
                    if(value>top1){top2=top1;top1=value;best=static_cast<int>(j);}
                    else if(value>top2)top2=value;
                }
                require(std::isfinite(maximum) && std::isfinite(logits[label]),
                        "qualitylogits label or normalizer nonfinite");
                double sum=0.0;
                for(const float value:logits)sum+=std::exp(static_cast<double>(value)-maximum);
                const double logsumexp=maximum+std::log(sum);
                const double nll=logsumexp-static_cast<double>(logits[label]);
                total_nll+=nll;
                csv << row << ',' << label << ',' << std::setprecision(12) << nll << ','
                    << best << ',' << (top1-top2) << ',' << logits[label] << ',' << logsumexp << '\n';
                if(row+1<scored_rows)context->decode(label);
            }
            csv.close();binary.close();
            require(csv.good() && binary.good(),"qualitylogits output write");
            const auto coherent_o_calls=
                Exl3CudaLinearWorkspace::process_coherent_o_k7_calls_for_test()-
                coherent_o_calls_before;
            const auto coherent_o_rows=
                Exl3CudaLinearWorkspace::process_coherent_o_k7_rows_for_test()-
                coherent_o_rows_before;
            std::cout<<"QUALITY_COHERENT_O_K7 enabled="<<coherent_o_k7
                     <<" calls="<<coherent_o_calls
                     <<" rows="<<coherent_o_rows<<'\n';
            require(coherent_o_k7 ? coherent_o_calls>0 &&
                    coherent_o_rows>=static_cast<std::uint64_t>(scored_rows-1)
                    : coherent_o_calls==0 && coherent_o_rows==0,
                    "quality coherent K7 O dispatch mismatch");
            const auto coherent_down_k7_calls=
                Exl3CudaLinearWorkspace::process_coherent_down_k7_calls_for_test()-
                coherent_down_k7_calls_before;
            const auto coherent_down_k7_rows=
                Exl3CudaLinearWorkspace::process_coherent_down_k7_rows_for_test()-
                coherent_down_k7_rows_before;
            std::cout<<"QUALITY_COHERENT_DOWN_K7 enabled="<<coherent_down_k7
                     <<" calls="<<coherent_down_k7_calls
                     <<" rows="<<coherent_down_k7_rows<<'\n';
            require(coherent_down_k7 ? coherent_down_k7_calls>0 &&
                    coherent_down_k7_rows>=static_cast<std::uint64_t>(scored_rows-1)
                    : coherent_down_k7_calls==0 && coherent_down_k7_rows==0,
                    "quality coherent K7 down dispatch mismatch");
            constexpr const char* coherent_wide_names[5]={
                "q","qkv","z","o","gate_up"};
            for(int operation=0;operation<5;++operation) {
                const auto calls=Exl3CudaLinearWorkspace::
                    coherent_wide_k6_calls_for_test(operation)-
                    coherent_wide_calls_before[operation];
                const auto rows=Exl3CudaLinearWorkspace::
                    coherent_wide_k6_rows_for_test(operation)-
                    coherent_wide_rows_before[operation];
                std::cout<<"QUALITY_COHERENT_WIDE_K6 op="
                         <<coherent_wide_names[operation]
                         <<" enabled="<<coherent_wide_k6
                         <<" calls="<<calls<<" rows="<<rows<<'\n';
                require(coherent_wide_k6 ? calls>0 && rows>=calls :
                        calls==0 && rows==0,
                    "quality coherent wide K6 dispatch mismatch");
            }
            std::cout << "QUALITY_LOGITS PASS prefix=" << prefix_rows
                      << " labels=" << scored_rows << " mean_nll="
                      << total_nll/scored_rows << '\n';
            return 0;
        }
        if (mode == "targetrequest") {
            if (targetrequest_context_mode) {
                run_targetrequest_contexts(*target, targetrequest_prefix,
                                           targetrequest_count, targetrequest_outputs,
                                           targetrequest_context_count,
                                           targetrequest_source_offset,
                                           targetrequest_summary_path,
                                           targetrequest_tokens_path,
                                           targetrequest_rounds_path);
                return 0;
            }
            const int count = targetrequest_count, outputs = targetrequest_outputs;
            const auto& prefix = targetrequest_prefix;
            const auto& summary_path = targetrequest_summary_path;
            const auto& tokens_path = targetrequest_tokens_path;
            const std::string graph_targetrequest_value =
                env("NINFER_E5A4_TARGETREQUEST_GRAPH");
            require(graph_targetrequest_value.empty() ||
                        graph_targetrequest_value == "0" ||
                        graph_targetrequest_value == "1",
                    "targetrequest graph flag must be 0 or 1");
            const bool graph_targetrequest = graph_targetrequest_value == "1";
            const auto decode_projection_profile_path =
                env("NINFER_E5A4_TARGETREQUEST_DECODE_PROJECTION_PROFILE");
            require(decode_projection_profile_path.empty() ||
                        (!graph_targetrequest &&
                         env("NINFER_EXL3_TARGET_PROJECTION_TIMING") == "1"),
                    "targetrequest decode projection profile requires eager projection timing");
            require(!graph_targetrequest || !targetrequest_exact_host_kv,
                    "targetrequest graph is incompatible with exact HostKV");
            const std::string graph_lifecycle_value =
                env("NINFER_E5A4_TARGETREQUEST_GRAPH_LIFECYCLE");
            require(graph_lifecycle_value.empty() ||
                        graph_lifecycle_value == "0" ||
                        graph_lifecycle_value == "1",
                    "targetrequest graph lifecycle flag must be 0 or 1");
            const bool graph_lifecycle = graph_lifecycle_value == "1";
            require(!graph_lifecycle || graph_targetrequest,
                    "targetrequest graph lifecycle requires graph admission");
            std::vector<std::int64_t> tokens; tokens.reserve(outputs);
            const auto fast_native_mia_target_prefill_fp16_before =
                Exl3CudaLinearWorkspace::process_fast_native_mia_target_prefill_fp16_calls_for_test();
            const auto prefill_shared_score_before =
                ninfer::exl3::exl3_prefill_attention_shared_score_global_snapshot();
            std::size_t free_before_context=0,device_total=0;
            std::size_t free_after_context=0,free_after_prefill=0;
            cuda_check(cudaMemGetInfo(&free_before_context,&device_total),
                       "targetrequest memory before context");
            const auto resident_begin = std::chrono::steady_clock::now();
            auto ctx = target->create_context(
                targetrequest_exact_host_kv || targetrequest_ordinary_fp16);
            const auto context_end = std::chrono::steady_clock::now();
            cuda_check(cudaMemGetInfo(&free_after_context,&device_total),
                       "targetrequest memory after context");
            if(!targetrequest_ordinary_fp16 && !targetrequest_exact_host_kv)
                require(ctx->try_enable_oscar_from_environment(), "targetrequest canonical OSCAR attachment");
            const auto fast_prefill_tiled_attention_prefill_before=
                exl3_fast_prefill_tiled_attention_calls_for_test();
            const auto fast_prefill_rows4_attention_prefill_before=
                exl3_fast_prefill_rows4_attention_calls_for_test();
            const auto fast_prefill_rows8_attention_prefill_before=
                exl3_fast_prefill_rows8_attention_calls_for_test();
            const auto fast_prefill_wmma_attention_prefill_before=
                exl3_fast_prefill_wmma_attention_calls_for_test();
            const auto fast_prefill_wmma32_attention_prefill_before=
                exl3_fast_prefill_wmma32_attention_calls_for_test();
            const auto fast_prefill_rows2_attention_prefill_before=
                exl3_fast_prefill_rows2_attention_calls_for_test();
            const auto ingest_targetrequest_prefix = [&](Exl3TextContext& context) {
                if (!targetrequest_exact_host_kv) {
                    ingest_prefix(context, prefix, CommitSink{});
                    return;
                }
                require(prefix.size() >= 16,
                        "targetrequest exact HostKV prefix requires initial16");
                context.prefill(std::span<const std::int64_t>(prefix.data(), 16));
                std::size_t first = 16;
                while (first < prefix.size()) {
                    const std::size_t remaining = prefix.size() - first;
                    const std::size_t rows = remaining >= 1024 ? 1024 :
                        remaining >= 128 ? 128 : remaining >= 32 ? 32 : 16;
                    require(rows <= remaining,
                            "targetrequest exact HostKV prefix decomposition");
                    context.append_exact_prefill_wide(
                        std::span<const std::int64_t>(prefix.data() + first, rows));
                    context.finish_exact_prefill();
                    first += rows;
                }
                require(context.position() == static_cast<int>(prefix.size()),
                        "targetrequest exact HostKV prefix position");
                cuda_check(cudaDeviceSynchronize(),
                           "targetrequest exact HostKV prefix completion");
            };
            const auto ingest_begin = std::chrono::steady_clock::now();
            ingest_targetrequest_prefix(*ctx);
            const auto ingest_end = std::chrono::steady_clock::now();
            if(!decode_projection_profile_path.empty())
                ctx->prepare_target_projection_timing();
            cuda_check(cudaMemGetInfo(&free_after_prefill,&device_total),
                       "targetrequest memory after completed prefill");
            std::cout << "TARGETREQUEST_MEMORY device_total_bytes=" << device_total
                      << " free_before_context_bytes=" << free_before_context
                      << " free_after_context_bytes=" << free_after_context
                      << " free_after_prefill_bytes=" << free_after_prefill << '\n';
            const auto fast_native_mia_target_prefill_fp16_calls =
                Exl3CudaLinearWorkspace::process_fast_native_mia_target_prefill_fp16_calls_for_test() -
                fast_native_mia_target_prefill_fp16_before;
            const auto prefill_shared_score_after =
                ninfer::exl3::exl3_prefill_attention_shared_score_global_snapshot();
            require(prefill_shared_score_after.launch_attempts >=
                        prefill_shared_score_before.launch_attempts &&
                    prefill_shared_score_after.row_attempts >=
                        prefill_shared_score_before.row_attempts &&
                    prefill_shared_score_after.global_score_bytes_eliminated >=
                        prefill_shared_score_before.global_score_bytes_eliminated,
                "targetrequest shared-score telemetry regressed");
            const auto prefill_shared_score_launches =
                prefill_shared_score_after.launch_attempts -
                prefill_shared_score_before.launch_attempts;
            const auto prefill_shared_score_rows =
                prefill_shared_score_after.row_attempts -
                prefill_shared_score_before.row_attempts;
            const auto prefill_shared_score_bytes =
                prefill_shared_score_after.global_score_bytes_eliminated -
                prefill_shared_score_before.global_score_bytes_eliminated;
            const auto prefill_shared_score_threads256_calls =
                prefill_shared_score_after.threads256_launch_attempts -
                prefill_shared_score_before.threads256_launch_attempts;
            const auto prefill_shared_score_parallel_softmax_calls =
                prefill_shared_score_after.parallel_softmax_launch_attempts -
                prefill_shared_score_before.parallel_softmax_launch_attempts;
            const auto prefill_shared_score_head_split256_calls =
                prefill_shared_score_after.head_split256_launch_attempts -
                prefill_shared_score_before.head_split256_launch_attempts;
            const auto prefill_shared_score_dimension_split256_calls =
                prefill_shared_score_after.dimension_split256_launch_attempts -
                prefill_shared_score_before.dimension_split256_launch_attempts;
            const auto fast_prefill_tiled_attention_prefill_calls=
                exl3_fast_prefill_tiled_attention_calls_for_test()-
                fast_prefill_tiled_attention_prefill_before;
            const auto fast_prefill_rows4_attention_prefill_calls=
                exl3_fast_prefill_rows4_attention_calls_for_test()-
                fast_prefill_rows4_attention_prefill_before;
            const auto fast_prefill_rows8_attention_prefill_calls=
                exl3_fast_prefill_rows8_attention_calls_for_test()-
                fast_prefill_rows8_attention_prefill_before;
            const auto fast_prefill_wmma_attention_prefill_calls=
                exl3_fast_prefill_wmma_attention_calls_for_test()-
                fast_prefill_wmma_attention_prefill_before;
            const auto fast_prefill_wmma32_attention_prefill_calls=
                exl3_fast_prefill_wmma32_attention_calls_for_test()-
                fast_prefill_wmma32_attention_prefill_before;
            const auto fast_prefill_rows2_attention_prefill_calls=
                exl3_fast_prefill_rows2_attention_calls_for_test()-
                fast_prefill_rows2_attention_prefill_before;
            tokens.push_back(sample_target(*ctx));
            const auto first_output = std::chrono::steady_clock::now();
            auto graph_capture_begin = first_output;
            auto graph_capture_end = first_output;
            std::unique_ptr<GraphCurrentStreamGuard> graph_stream_guard;
            cudaStream_t graph_stream = nullptr;
            const int graph_class = env_int(
                "NINFER_E5A4_TARGETREQUEST_GRAPH_CLASS", 32);
            if (graph_targetrequest) {
                require(count >= 1024 && count + outputs - 1 <= 8192,
                        "targetrequest graph requires one stable S32 context class");
                require(targetrequest_ordinary_fp16 || graph_class == 32,
                        "targetrequest OSCAR graph requires graph class 32");
                graph_capture_begin = std::chrono::steady_clock::now();
                graph_stream_guard = std::make_unique<GraphCurrentStreamGuard>();
                graph_stream = graph_stream_guard->stream;
                cuda_check(cudaDeviceSynchronize(),
                           "targetrequest graph pre-capture sync");
                if(!targetrequest_ordinary_fp16) {
                    ctx->oscar_set_graph_class(graph_class);
                    ctx->oscar_sync_device_state(graph_stream);
                }
                cuda_check(cudaStreamSynchronize(graph_stream),
                           "targetrequest graph state setup sync");
                // Capture the result before composing the diagnostic.  The
                // two function arguments otherwise have unspecified
                // evaluation order, which can read graph_status() before the
                // capture call updates it and report the stale reset sentinel.
                const bool graph_captured = ctx->capture_decode_graph(graph_stream);
                require(graph_captured,
                        "targetrequest graph capture failed: " + ctx->graph_status());
                cuda_check(cudaStreamSynchronize(graph_stream),
                           "targetrequest graph capture sync");
                graph_capture_end = std::chrono::steady_clock::now();
            }
            const auto target_m1_k6_n16_before=
                Exl3CudaLinearWorkspace::process_target_m1_k6_n16_calls_for_test();
            const auto target_m1_k7_three_word_before=
                Exl3CudaLinearWorkspace::process_target_m1_k7_three_word_calls_for_test();
            const auto target_k5_small_m_batch_before=
                ctx->target_k5_small_m_batch_calls();
            const auto native_k6_critical_path_before=
                ctx->native_k6_critical_path_calls();
            const auto native_k6_register_pipeline_before=
                ctx->native_k6_register_pipeline_calls();
            const auto fast_same_weights_fp16_accum_before=
                Exl3CudaLinearWorkspace::process_fast_same_weights_fp16_accum_calls_for_test();
            const auto fast_same_weights_fp16_m1_before=
                Exl3CudaLinearWorkspace::process_fast_same_weights_fp16_m1_calls_for_test();
            const auto fast_same_weights_fp16_m1_wide_n32_before=
                Exl3CudaLinearWorkspace::process_fast_same_weights_fp16_m1_wide_n32_calls_for_test();
            const auto fast_same_weights_fp16_m1_n16_before=
                Exl3CudaLinearWorkspace::process_fast_same_weights_fp16_m1_n16_calls_for_test();
            const auto fast_native_persistent_m1_before=
                Exl3CudaLinearWorkspace::process_fast_native_persistent_m1_calls_for_test();
            const auto fast_native_mia_m1_fp16_before=
                Exl3CudaLinearWorkspace::process_fast_native_mia_m1_fp16_calls_for_test();
            const auto fast_same_weights_fp16kv_m1_gate_up_pair_before=
                Exl3CudaLinearWorkspace::process_fast_same_weights_fp16kv_m1_gate_up_pair_calls_for_test();
            const auto fast_same_weights_fp16kv_m1_kv_pair_before=
                Exl3CudaLinearWorkspace::process_fast_same_weights_fp16kv_m1_kv_pair_calls_for_test();
            const auto fast_same_weights_fp16_m1_n64_before=
                Exl3CudaLinearWorkspace::process_fast_same_weights_fp16_m1_n64_calls_for_test();
            const auto fast_same_weights_fp16_m1_n64_k5_before=
                Exl3CudaLinearWorkspace::process_fast_same_weights_fp16_m1_n64_k5_calls_for_test();
            const auto fast_same_weights_int8_gemv_before=
                Exl3CudaLinearWorkspace::process_fast_same_weights_int8_gemv_calls_for_test();
            const auto fast_same_weights_int8_gemv_k7_before=
                Exl3CudaLinearWorkspace::process_fast_same_weights_int8_gemv_k7_calls_for_test();
            const auto fast_same_weights_int8_gemv_down_k6_before=
                Exl3CudaLinearWorkspace::process_fast_same_weights_int8_gemv_down_k6_calls_for_test();
            const auto fast_same_weights_int8_gemv_down_k7_before=
                Exl3CudaLinearWorkspace::process_fast_same_weights_int8_gemv_down_k7_calls_for_test();
            const auto fast_fused_flash_attention_before=
                exl3_fast_fused_flash_attention_calls_for_test();
            const auto fast_whole_context_fused_attention_before=
                exl3_fast_whole_context_fused_attention_calls_for_test();
            const auto fast_prefill_tiled_attention_before=
                exl3_fast_prefill_tiled_attention_calls_for_test();
            const auto fast_prefill_rows4_attention_before=
                exl3_fast_prefill_rows4_attention_calls_for_test();
            const auto fast_prefill_rows8_attention_before=
                exl3_fast_prefill_rows8_attention_calls_for_test();
            const auto fast_prefill_wmma_attention_before=
                exl3_fast_prefill_wmma_attention_calls_for_test();
            const auto fast_prefill_wmma32_attention_before=
                exl3_fast_prefill_wmma32_attention_calls_for_test();
            const auto fast_prefill_rows2_attention_before=
                exl3_fast_prefill_rows2_attention_calls_for_test();
            const auto fast_cublas_attention_before=
                exl3_fast_cublas_attention_calls_for_test();
            const auto fast_online_decode_attention_before=
                exl3_fast_online_decode_attention_calls_for_test();
            const auto fast_gdn_decode_conv_before=
                ctx->fast_same_weights_fp16kv_gdn_decode_conv_calls();
            const auto exact_attention_scalar_before=
                ctx->exact_attention_gqa_six_softmax_fused_scalar_values_calls();
            const auto exact_attention_scalar_rows_before=
                ctx->exact_attention_gqa_six_softmax_fused_scalar_values_rows();
            const auto exact_attention_decode_fused_before=
                ctx->exact_attention_gqa_six_decode_fused_calls();
            const auto decode_begin = std::chrono::steady_clock::now();
            for (int i = 1; i < outputs; ++i) {
                if(i==1 && !decode_projection_profile_path.empty())
                    ctx->begin_target_projection_timing_round(1);
                if (graph_targetrequest) ctx->decode_graph(tokens.back(), graph_stream);
                else ctx->decode(tokens.back());
                tokens.push_back(sample_target(*ctx, graph_stream));
                if(i==1 && !decode_projection_profile_path.empty()) {
                    cuda_check(cudaDeviceSynchronize(),
                        "targetrequest decode projection profile completion");
                    const auto records=
                        ctx->finish_target_projection_timing_round_after_synchronize();
                    require(records.size()==Exl3TargetProjectionTiming::kGroupsPerPass,
                        "targetrequest decode projection inventory");
                    std::ofstream projections(decode_projection_profile_path);
                    require(projections.good(),
                        "targetrequest decode projection output open");
                    projections.imbue(std::locale::classic());
                    projections<<"layer,operator,rows,K,in_features,out_features,topology,calls,microseconds\n";
                    for(const auto& record:records)
                        projections<<record.layer<<','<<
                            target_projection_operator_name(record.operation)<<','<<
                            record.rows<<','<<record.K<<','<<
                            record.in_features<<','<<record.out_features<<','<<
                            target_projection_topology_name(record.topology)<<','<<
                            record.calls<<','<<std::fixed<<std::setprecision(3)<<
                            record.microseconds<<'\n';
                    projections.flush();
                    require(projections.good(),
                        "targetrequest decode projection output write");
                    std::cout<<"TARGETREQUEST_DECODE_PROJECTION_PROFILE PASS rows="<<
                        records.size()<<'\n';
                }
            }
            const auto last_output = std::chrono::steady_clock::now();
            if(env("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_GDN_M1_GATE_UP_PAIR")=="1") {
                const auto submissions=
                    ctx->fast_same_weights_fp16kv_gdn_m1_gate_up_pair_submissions();
                std::cout<<"TARGETREQUEST_GDN_M1_GATE_UP_PAIR submissions="<<
                    submissions<<" submissions_include_capture=1 graph_replays_not_counted=1\n";
                require(submissions>0,
                    "targetrequest GDN M1 gate/up pair flag did not dispatch");
            }
            const auto target_m1_k6_n16_calls=
                Exl3CudaLinearWorkspace::process_target_m1_k6_n16_calls_for_test()-
                target_m1_k6_n16_before;
            const auto target_m1_k7_three_word_calls=
                Exl3CudaLinearWorkspace::process_target_m1_k7_three_word_calls_for_test()-
                target_m1_k7_three_word_before;
            const auto target_k5_small_m_batch_calls=
                ctx->target_k5_small_m_batch_calls()-target_k5_small_m_batch_before;
            const auto native_k6_critical_path_calls=
                ctx->native_k6_critical_path_calls()-
                native_k6_critical_path_before;
            const auto native_k6_register_pipeline_calls=
                ctx->native_k6_register_pipeline_calls()-
                native_k6_register_pipeline_before;
            const auto fast_same_weights_fp16_accum_calls=
                Exl3CudaLinearWorkspace::process_fast_same_weights_fp16_accum_calls_for_test()-
                fast_same_weights_fp16_accum_before;
            const auto fast_same_weights_fp16_m1_calls=
                Exl3CudaLinearWorkspace::process_fast_same_weights_fp16_m1_calls_for_test()-
                fast_same_weights_fp16_m1_before;
            const auto fast_same_weights_fp16_m1_wide_n32_calls=
                Exl3CudaLinearWorkspace::process_fast_same_weights_fp16_m1_wide_n32_calls_for_test()-
                fast_same_weights_fp16_m1_wide_n32_before;
            const auto fast_same_weights_fp16_m1_n16_calls=
                Exl3CudaLinearWorkspace::process_fast_same_weights_fp16_m1_n16_calls_for_test()-
                fast_same_weights_fp16_m1_n16_before;
            const auto fast_native_persistent_m1_calls=
                Exl3CudaLinearWorkspace::process_fast_native_persistent_m1_calls_for_test()-
                fast_native_persistent_m1_before;
            const auto fast_native_mia_m1_fp16_calls=
                Exl3CudaLinearWorkspace::process_fast_native_mia_m1_fp16_calls_for_test()-
                fast_native_mia_m1_fp16_before;
            const auto fast_same_weights_fp16kv_m1_gate_up_pair_calls=
                Exl3CudaLinearWorkspace::process_fast_same_weights_fp16kv_m1_gate_up_pair_calls_for_test()-
                fast_same_weights_fp16kv_m1_gate_up_pair_before;
            const auto fast_same_weights_fp16kv_m1_kv_pair_calls=
                Exl3CudaLinearWorkspace::process_fast_same_weights_fp16kv_m1_kv_pair_calls_for_test()-
                fast_same_weights_fp16kv_m1_kv_pair_before;
            const auto fast_same_weights_fp16_m1_n64_calls=
                Exl3CudaLinearWorkspace::process_fast_same_weights_fp16_m1_n64_calls_for_test()-
                fast_same_weights_fp16_m1_n64_before;
            const auto fast_same_weights_fp16_m1_n64_k5_calls=
                Exl3CudaLinearWorkspace::process_fast_same_weights_fp16_m1_n64_k5_calls_for_test()-
                fast_same_weights_fp16_m1_n64_k5_before;
            const auto fast_same_weights_int8_gemv_calls=
                Exl3CudaLinearWorkspace::process_fast_same_weights_int8_gemv_calls_for_test()-
                fast_same_weights_int8_gemv_before;
            const auto fast_same_weights_int8_gemv_k7_calls=
                Exl3CudaLinearWorkspace::process_fast_same_weights_int8_gemv_k7_calls_for_test()-
                fast_same_weights_int8_gemv_k7_before;
            const auto fast_same_weights_int8_gemv_down_k6_calls=
                Exl3CudaLinearWorkspace::process_fast_same_weights_int8_gemv_down_k6_calls_for_test()-
                fast_same_weights_int8_gemv_down_k6_before;
            const auto fast_same_weights_int8_gemv_down_k7_calls=
                Exl3CudaLinearWorkspace::process_fast_same_weights_int8_gemv_down_k7_calls_for_test()-
                fast_same_weights_int8_gemv_down_k7_before;
            const auto fast_fused_flash_attention_calls=
                exl3_fast_fused_flash_attention_calls_for_test()-
                fast_fused_flash_attention_before;
            const auto fast_whole_context_fused_attention_calls=
                exl3_fast_whole_context_fused_attention_calls_for_test()-
                fast_whole_context_fused_attention_before;
            const auto fast_prefill_tiled_attention_calls=
                exl3_fast_prefill_tiled_attention_calls_for_test()-
                fast_prefill_tiled_attention_before;
            const auto fast_prefill_rows4_attention_calls=
                exl3_fast_prefill_rows4_attention_calls_for_test()-
                fast_prefill_rows4_attention_before;
            const auto fast_prefill_rows8_attention_calls=
                exl3_fast_prefill_rows8_attention_calls_for_test()-
                fast_prefill_rows8_attention_before;
            const auto fast_prefill_wmma_attention_calls=
                exl3_fast_prefill_wmma_attention_calls_for_test()-
                fast_prefill_wmma_attention_before;
            const auto fast_prefill_wmma32_attention_calls=
                exl3_fast_prefill_wmma32_attention_calls_for_test()-
                fast_prefill_wmma32_attention_before;
            const auto fast_prefill_rows2_attention_calls=
                exl3_fast_prefill_rows2_attention_calls_for_test()-
                fast_prefill_rows2_attention_before;
            const auto fast_cublas_attention_calls=
                exl3_fast_cublas_attention_calls_for_test()-
                fast_cublas_attention_before;
            const auto fast_online_decode_attention_calls=
                exl3_fast_online_decode_attention_calls_for_test()-
                fast_online_decode_attention_before;
            const auto fast_gdn_decode_conv_calls=
                ctx->fast_same_weights_fp16kv_gdn_decode_conv_calls()-
                fast_gdn_decode_conv_before;
            const auto fast_wide_prefill_gemm_calls=
                ctx->fast_wide_prefill_gemm_calls();
            const auto fast_wide_prefill_gemm_rows=
                ctx->fast_wide_prefill_gemm_rows();
            const auto exact_attention_scalar_calls=
                ctx->exact_attention_gqa_six_softmax_fused_scalar_values_calls()-
                exact_attention_scalar_before;
            const auto exact_attention_scalar_rows=
                ctx->exact_attention_gqa_six_softmax_fused_scalar_values_rows()-
                exact_attention_scalar_rows_before;
            const auto exact_attention_decode_fused_calls=
                ctx->exact_attention_gqa_six_decode_fused_calls()-
                exact_attention_decode_fused_before;
            require(static_cast<std::int64_t>(ctx->position()) == static_cast<std::int64_t>(count) + outputs - 1 && tokens.size() == static_cast<std::size_t>(outputs),
                    "targetrequest pending final token/count mismatch");
            const int terminal_position = ctx->position();
            if (graph_lifecycle) {
                const auto run_lifecycle = [&](Exl3TextContext& lifecycle_context,
                                               bool reset_first,
                                               const std::string& label) {
                    if (reset_first) {
                        lifecycle_context.reset(graph_stream);
                        cuda_check(cudaStreamSynchronize(graph_stream),
                                   (label + " reset sync").c_str());
                    }
                    require(!lifecycle_context.graph_active(),
                            label + " must require capture");
                    ingest_targetrequest_prefix(lifecycle_context);
                    std::vector<std::int64_t> replay_tokens;
                    replay_tokens.reserve(outputs);
                    replay_tokens.push_back(sample_target(lifecycle_context, graph_stream));
                    if(!targetrequest_ordinary_fp16) {
                        lifecycle_context.oscar_set_graph_class(graph_class);
                        lifecycle_context.oscar_sync_device_state(graph_stream);
                    }
                    cuda_check(cudaStreamSynchronize(graph_stream),
                               (label + " graph state setup sync").c_str());
                    require(lifecycle_context.capture_decode_graph(graph_stream),
                            label + " capture failed: " + lifecycle_context.graph_status());
                    cuda_check(cudaStreamSynchronize(graph_stream),
                               (label + " capture sync").c_str());
                    require(lifecycle_context.graph_active(),
                            label + " capture did not activate graph");
                    for (int i = 1; i < outputs; ++i) {
                        lifecycle_context.decode_graph(replay_tokens.back(), graph_stream);
                        replay_tokens.push_back(sample_target(lifecycle_context, graph_stream));
                    }
                    require(replay_tokens == tokens,
                            label + " token mismatch");
                    require(lifecycle_context.position() == count + outputs - 1,
                            label + " position mismatch");
                };
                run_lifecycle(*ctx, true, "targetrequest graph reset-recapture");
                cuda_check(cudaStreamSynchronize(graph_stream),
                           "targetrequest graph pre-destroy sync");
                ctx.reset();
                auto boundary_context = target->create_context(
                    targetrequest_exact_host_kv || targetrequest_ordinary_fp16);
                if(!targetrequest_ordinary_fp16)
                    require(boundary_context->try_enable_oscar_from_environment(),
                            "targetrequest graph boundary OSCAR attachment");
                bool pre_capture_rejected = false;
                try { boundary_context->decode_graph(tokens.front(), graph_stream); }
                catch (const std::exception&) { pre_capture_rejected = true; }
                require(pre_capture_rejected,
                        "targetrequest graph fresh context replay admitted before capture");
                run_lifecycle(*boundary_context, false,
                              "targetrequest graph context-boundary");
                cuda_check(cudaStreamSynchronize(graph_stream),
                           "targetrequest graph boundary completion sync");
            }
            const auto ms = [](auto a, auto z) { return std::chrono::duration<double, std::milli>(z-a).count(); };
            std::ostringstream ids;
            std::ofstream token_file(tokens_path); token_file.imbue(std::locale::classic());
            token_file << "ordinal,token_id\n";
            for (std::size_t i=0; i<tokens.size(); ++i) {
                if (i) ids << ';'; ids << tokens[i]; token_file << i << ',' << tokens[i] << '\n';
            }
            token_file.flush(); require(token_file.good(), "targetrequest token output failed");
            const bool skip_graph_export =
                env("NINFER_E5A4_SKIP_GRAPH_EXPORT") == "1";
            std::string terminal_state_hash;
            if (skip_graph_export) {
                require(graph_targetrequest && ctx->graph_active(),
                        "graph export skip requires an admitted active graph");
                terminal_state_hash = "GRAPH_ACTIVE_EXPORT_SKIPPED";
            } else {
                const auto terminal_state=ctx->export_exact_host_state();
                terminal_state_hash = std::to_string(
                    terminal_state->represented_payload_hash_for_test());
            }
            const auto host_kv_gdn_graph_stats=
                ctx->host_kv_gdn_segment_graph_stats();
            const auto host_kv_full_layer_graph_stats=
                ctx->host_kv_full_layer_graph_stats();
            const auto ordinary_full_layer_graph_stats=
                ctx->ordinary_full_layer_graph_stats();
            const auto host_kv_mlp_tail_graph_stats=
                ctx->host_kv_mlp_tail_graph_stats();
            const auto numeric_prefill_stats =
                ctx->numeric_prefill_projection_stats();
            std::ofstream summary(summary_path); summary.imbue(std::locale::classic());
             summary << "mode,execution,kv_mode,ingested,emitted,target_decodes,position,pending_token,state_payload_hash,target_m1_k6_n16_calls,target_m1_k7_three_word_calls,target_k5_small_m_batch_calls,native_k6_critical_path_calls,native_k6_register_pipeline_calls,fast_same_weights_fp16_accum_calls,fast_same_weights_fp16_m1_calls,fast_same_weights_fp16_m1_wide_n32_calls,fast_same_weights_fp16_m1_n16_calls,fast_native_persistent_m1_calls,fast_native_mia_target_prefill_fp16_calls,prefill_shared_score_launches,prefill_shared_score_rows,prefill_shared_score_bytes,prefill_shared_score_threads256_calls,prefill_shared_score_parallel_softmax_calls,prefill_shared_score_head_split256_calls,prefill_shared_score_dimension_split256_calls,fast_same_weights_fp16kv_m1_gate_up_pair_calls,fast_same_weights_fp16kv_m1_kv_pair_calls,fast_same_weights_fp16_m1_n64_calls,fast_same_weights_fp16_m1_n64_k5_calls,fast_same_weights_int8_gemv_calls,fast_same_weights_int8_gemv_k7_calls,fast_same_weights_int8_gemv_down_k6_calls,fast_same_weights_int8_gemv_down_k7_calls,fast_fused_flash_attention_calls,fast_whole_context_fused_attention_calls,fast_prefill_tiled_attention_calls,fast_prefill_tiled_attention_prefill_calls,fast_prefill_rows4_attention_calls,fast_prefill_rows4_attention_prefill_calls,fast_prefill_rows8_attention_calls,fast_prefill_rows8_attention_prefill_calls,fast_prefill_wmma_attention_calls,fast_prefill_wmma_attention_prefill_calls,fast_prefill_wmma32_attention_calls,fast_prefill_wmma32_attention_prefill_calls,fast_prefill_rows2_attention_calls,fast_prefill_rows2_attention_prefill_calls,fast_online_decode_attention_calls,fast_cublas_attention_calls,fast_same_weights_fp16kv_gdn_decode_conv_calls,fast_wide_prefill_gemm_calls,fast_wide_prefill_gemm_rows,exact_attention_scalar_calls,exact_attention_scalar_rows,exact_attention_gqa_six_decode_fused_calls,numeric_prefill_workspace_bytes,numeric_prefill_calls,numeric_prefill_k6_calls,numeric_prefill_k7_calls,numeric_prefill_rows,numeric_prefill_fused_original_calls,numeric_prefill_fused_original_rows,numeric_prefill_fp16_compute_calls,numeric_prefill_fp16_compute_rows,numeric_prefill_packed_direct_k6_calls,numeric_prefill_packed_direct_k6_rows,numeric_prefill_mia_prefill_fp16_calls,numeric_prefill_mia_prefill_fp16_rows,executable_main_first_token_available_ms,executable_main_last_token_available_ms,resident_first_token_available_ms,resident_last_token_available_ms,context_create_ms,oscar_attach_ms,ingestion_wall_ms,seed_wall_ms,graph_capture_ms,subsequent_decode_wall_ms,decode_execution_wall_ms,lifecycle_pass,host_kv_gdn_segment_graph_captures,host_kv_gdn_segment_graph_replays,host_kv_gdn_segment_graph_capture_ms,host_kv_full_layer_graph_captures,host_kv_full_layer_graph_replays,host_kv_full_layer_graph_capture_ms,host_kv_full_layer_graph_six_softmax_triple_captures,host_kv_full_layer_graph_k6_stream_reduction_captures,host_kv_full_layer_graph_extended_stream_reduction_captures,host_kv_full_layer_graph_target_down_k6_async_a_captures,host_kv_full_layer_graph_target_k6_small_m_async_a_captures,host_kv_full_layer_graph_target_k7_small_m_async_a_captures,ordinary_full_layer_graph_captures,ordinary_full_layer_graph_replays,ordinary_full_layer_graph_capture_ms,host_kv_mlp_tail_graph_captures,host_kv_mlp_tail_graph_replays,host_kv_mlp_tail_graph_capture_ms,token_ids\n";
            summary << std::fixed << std::setprecision(9) << "targetrequest,"
                    << (graph_targetrequest ? "graph" : "eager") << ','
                    << (targetrequest_ordinary_fp16 ? "ordinary_fp16_device" :
                        targetrequest_exact_host_kv ? "exact_host" : "oscar") << ','
                    << count << ',' << tokens.size() << ','
                    << outputs-1 << ',' << terminal_position << ',' << tokens.back() << ','
                    << terminal_state_hash << ',' << target_m1_k6_n16_calls << ','
                    << target_m1_k7_three_word_calls << ','
                    << target_k5_small_m_batch_calls << ','
                    << native_k6_critical_path_calls << ','
                    << native_k6_register_pipeline_calls << ','
                    << fast_same_weights_fp16_accum_calls << ','
                    << fast_same_weights_fp16_m1_calls << ','
                    << fast_same_weights_fp16_m1_wide_n32_calls << ','
                    << fast_same_weights_fp16_m1_n16_calls << ','
                    << fast_native_persistent_m1_calls << ','
                    << fast_native_mia_target_prefill_fp16_calls << ','
                    << prefill_shared_score_launches << ','
                    << prefill_shared_score_rows << ','
                    << prefill_shared_score_bytes << ','
                    << prefill_shared_score_threads256_calls << ','
                    << prefill_shared_score_parallel_softmax_calls << ','
                    << prefill_shared_score_head_split256_calls << ','
                    << prefill_shared_score_dimension_split256_calls << ','
                    << fast_same_weights_fp16kv_m1_gate_up_pair_calls << ','
                    << fast_same_weights_fp16kv_m1_kv_pair_calls << ','
                    << fast_same_weights_fp16_m1_n64_calls << ','
                    << fast_same_weights_fp16_m1_n64_k5_calls << ','
                    << fast_same_weights_int8_gemv_calls << ','
                    << fast_same_weights_int8_gemv_k7_calls << ','
                    << fast_same_weights_int8_gemv_down_k6_calls << ','
                    << fast_same_weights_int8_gemv_down_k7_calls << ','
                    << fast_fused_flash_attention_calls << ','
                     << fast_whole_context_fused_attention_calls << ','
                     << fast_prefill_tiled_attention_calls << ','
                     << fast_prefill_tiled_attention_prefill_calls << ','
                     << fast_prefill_rows4_attention_calls << ','
                     << fast_prefill_rows4_attention_prefill_calls << ','
                     << fast_prefill_rows8_attention_calls << ','
                     << fast_prefill_rows8_attention_prefill_calls << ','
                     << fast_prefill_wmma_attention_calls << ','
                     << fast_prefill_wmma_attention_prefill_calls << ','
                     << fast_prefill_wmma32_attention_calls << ','
                     << fast_prefill_wmma32_attention_prefill_calls << ','
                     << fast_prefill_rows2_attention_calls << ','
                      << fast_prefill_rows2_attention_prefill_calls << ','
                       << fast_online_decode_attention_calls << ','
                       << fast_cublas_attention_calls << ','
                      << fast_gdn_decode_conv_calls << ','
                    << fast_wide_prefill_gemm_calls << ','
                    << fast_wide_prefill_gemm_rows << ','
                     << exact_attention_scalar_calls << ','
                     << exact_attention_scalar_rows << ','
                     << exact_attention_decode_fused_calls << ','
                     << numeric_prefill_stats.workspace_bytes << ','
                    << numeric_prefill_stats.calls << ','
                    << numeric_prefill_stats.k6_calls << ','
                    << numeric_prefill_stats.k7_calls << ','
                    << numeric_prefill_stats.rows << ','
                    << numeric_prefill_stats.fused_original_calls << ','
                    << numeric_prefill_stats.fused_original_rows << ','
                    << numeric_prefill_stats.fp16_compute_calls << ','
                    << numeric_prefill_stats.fp16_compute_rows << ','
                     << numeric_prefill_stats.packed_direct_k6_calls << ','
                     << numeric_prefill_stats.packed_direct_k6_rows << ','
                     << numeric_prefill_stats.mia_prefill_fp16_calls << ','
                     << numeric_prefill_stats.mia_prefill_fp16_rows << ','
                    << ms(entry_stamp.monotonic,first_output) << ',' << ms(entry_stamp.monotonic,last_output) << ','
                    << ms(resident_begin,first_output) << ',' << ms(resident_begin,last_output) << ','
                    << ms(resident_begin,context_end) << ',' << ms(context_end,ingest_begin) << ','
                    << ms(ingest_begin,ingest_end) << ',' << ms(ingest_end,first_output) << ','
                    << ms(graph_capture_begin,graph_capture_end) << ','
                    << (outputs == 1 ? 0.0 : ms(first_output,last_output)) << ','
                    << (outputs == 1 ? 0.0 : ms(decode_begin,last_output)) << ','
                    << (graph_lifecycle ? 1 : 0) << ','
                    << host_kv_gdn_graph_stats.captures << ','
                    << host_kv_gdn_graph_stats.replays << ','
                    << host_kv_gdn_graph_stats.capture_ms << ','
                    << host_kv_full_layer_graph_stats.captures << ','
                    << host_kv_full_layer_graph_stats.replays << ','
                    << host_kv_full_layer_graph_stats.capture_ms << ','
                    << host_kv_full_layer_graph_stats.six_softmax_triple_captures << ','
                    << host_kv_full_layer_graph_stats.k6_stream_reduction_captures << ','
                    << host_kv_full_layer_graph_stats.extended_stream_reduction_captures << ','
                     << host_kv_full_layer_graph_stats.target_down_k6_async_a_captures << ','
                     << host_kv_full_layer_graph_stats.target_k6_small_m_async_a_captures << ','
                     << host_kv_full_layer_graph_stats.target_k7_small_m_async_a_captures << ','
                     << ordinary_full_layer_graph_stats.captures << ','
                     << ordinary_full_layer_graph_stats.replays << ','
                     << ordinary_full_layer_graph_stats.capture_ms << ','
                     << host_kv_mlp_tail_graph_stats.captures << ','
                    << host_kv_mlp_tail_graph_stats.replays << ','
                    << host_kv_mlp_tail_graph_stats.capture_ms << ','
                    << ids.str() << '\n';
            summary.flush(); require(summary.good(), "targetrequest summary output failed");
            std::cout << "TARGETREQUEST_RECON reconstructed_weight_calls="
                      << numeric_prefill_stats.reconstructed_weight_calls
                      << " reconstructed_weight_bytes="
                      << numeric_prefill_stats.reconstructed_weight_bytes
                      << " reused_weight_calls="
                      << numeric_prefill_stats.reused_weight_calls
                      << " reused_weight_bytes="
                      << numeric_prefill_stats.reused_weight_bytes
                      << " mxfp8_calls=" << numeric_prefill_stats.mxfp8_calls
                      << " mxfp8_rows=" << numeric_prefill_stats.mxfp8_rows
                      << " large_lt_calls="
                      << numeric_prefill_stats.large_lt_calls
                      << " cached_weight_capacity_bytes="
                      << numeric_prefill_stats.cached_weight_capacity_bytes << '\n';
            std::cout << "TARGETREQUEST_GDN_PACKED_K5 enabled="
                      << (env("NINFER_EXL3_FAST_GDN_BULK_MLP_PACKED_K5")=="1")
                      << " gate_up_calls="
                      << numeric_prefill_stats.packed_direct_k5_gate_up_calls
                      << " down_calls="
                      << numeric_prefill_stats.packed_direct_k5_down_calls
                      << " rows="
                      << numeric_prefill_stats.packed_direct_k5_rows << '\n';
            std::cout << "TARGETREQUEST PASS ingested=" << count << " emitted=" << outputs
                      << " target_decodes=" << outputs-1 << " pending=" << tokens.back()
                      << " target_down_k6_async_a_calls="
                      << ctx->target_down_k6_async_a_calls()
                      << " target_m1_k6_n16_calls=" << target_m1_k6_n16_calls
                      << " target_m1_k7_three_word_calls="
                      << target_m1_k7_three_word_calls
                      << " fast_same_weights_fp16_m1_calls="
                      << fast_same_weights_fp16_m1_calls
                      << " fast_same_weights_fp16_m1_n64_calls="
                      << fast_same_weights_fp16_m1_n64_calls
                      << " fast_same_weights_fp16_m1_n64_k5_calls="
                      << fast_same_weights_fp16_m1_n64_k5_calls
                      << " fast_native_mia_m1_fp16_calls="
                      << fast_native_mia_m1_fp16_calls
                      << " fast_native_mia_target_prefill_fp16_calls="
                      << fast_native_mia_target_prefill_fp16_calls
                      << " fast_same_weights_int8_gemv_k7_calls="
                      << fast_same_weights_int8_gemv_k7_calls
                      << " fast_same_weights_int8_gemv_down_k6_calls="
                      << fast_same_weights_int8_gemv_down_k6_calls
                      << " fast_same_weights_int8_gemv_down_k7_calls="
                      << fast_same_weights_int8_gemv_down_k7_calls
                      << " fast_fused_flash_attention_calls="
                      << fast_fused_flash_attention_calls
                      << " fast_whole_context_fused_attention_calls="
                      << fast_whole_context_fused_attention_calls
                      << " fast_prefill_tiled_attention_calls="
                      << fast_prefill_tiled_attention_calls
                      << " fast_prefill_rows4_attention_calls="
                      << fast_prefill_rows4_attention_calls
                      << " fast_prefill_rows8_attention_calls="
                      << fast_prefill_rows8_attention_calls
                      << " fast_prefill_wmma_attention_calls="
                      << fast_prefill_wmma_attention_calls
                      << " fast_prefill_wmma32_attention_calls="
                      << fast_prefill_wmma32_attention_calls
                      << " fast_prefill_rows2_attention_calls="
                      << fast_prefill_rows2_attention_calls
                      << " fast_online_decode_attention_calls="
                      << fast_online_decode_attention_calls
                      << " fast_cublas_attention_calls="
                      << fast_cublas_attention_calls
                      << " fast_same_weights_fp16kv_gdn_decode_conv_calls="
                      << fast_gdn_decode_conv_calls
                      << " fast_wide_prefill_gemm_calls="
                      << fast_wide_prefill_gemm_calls
                      << " fast_wide_prefill_gemm_rows="
                      << fast_wide_prefill_gemm_rows
                      << " numeric_prefill_persistent_prefill_calls="
                      << numeric_prefill_stats.persistent_prefill_calls
                      << " numeric_prefill_persistent_prefill_rows="
                      << numeric_prefill_stats.persistent_prefill_rows
                      << " exact_attention_scalar_calls="
                      << exact_attention_scalar_calls
                      << " exact_attention_scalar_rows="
                      << exact_attention_scalar_rows
                      << " mlp_tail_graph_captures="
                      << host_kv_mlp_tail_graph_stats.captures
                      << " mlp_tail_graph_replays="
                      << host_kv_mlp_tail_graph_stats.replays
                      << " ordinary_full_layer_graph_captures="
                      << ordinary_full_layer_graph_stats.captures
                      << " ordinary_full_layer_graph_replays="
                      << ordinary_full_layer_graph_stats.replays
                      << "\nTARGET_REQUEST_DONE\n";
            const bool coherent_o_k7=env("NINFER_EXL3_COHERENT_O_K7")=="1";
            const auto coherent_o_calls=Exl3CudaLinearWorkspace::
                process_coherent_o_k7_calls_for_test();
            const auto coherent_o_rows=Exl3CudaLinearWorkspace::
                process_coherent_o_k7_rows_for_test();
            std::cout<<"TARGETREQUEST_COHERENT_O_K7 enabled="<<coherent_o_k7
                     <<" calls="<<coherent_o_calls
                     <<" rows="<<coherent_o_rows<<'\n';
            require(!coherent_o_k7 ||
                (coherent_o_calls>0 && coherent_o_rows>=static_cast<std::uint64_t>(outputs-1)),
                "coherent K7 O did not dispatch during target-only decode");
            const bool coherent_down_k7=env("NINFER_EXL3_COHERENT_DOWN_K7")=="1";
            const auto coherent_down_k7_calls=Exl3CudaLinearWorkspace::
                process_coherent_down_k7_calls_for_test();
            const auto coherent_down_k7_rows=Exl3CudaLinearWorkspace::
                process_coherent_down_k7_rows_for_test();
            std::cout<<"TARGETREQUEST_COHERENT_DOWN_K7 enabled="<<coherent_down_k7
                     <<" calls="<<coherent_down_k7_calls
                     <<" rows="<<coherent_down_k7_rows<<'\n';
            require(coherent_down_k7 ? coherent_down_k7_calls>0 &&
                    coherent_down_k7_rows>=static_cast<std::uint64_t>(outputs-1)
                    : coherent_down_k7_calls==0 && coherent_down_k7_rows==0,
                "coherent K7 down target-only dispatch mismatch");
            const bool coherent_wide_k6=
                env("NINFER_EXL3_TARGET_COHERENT_WIDE_K6")=="1";
            const bool coherent_wide_split10=
                env("NINFER_EXL3_TARGET_COHERENT_WIDE_K6_SPLIT10")=="1";
            constexpr const char* coherent_wide_names[5]={
                "q","qkv","z","o","gate_up"};
            for(int operation=0;operation<5;++operation) {
                const auto calls=Exl3CudaLinearWorkspace::
                    coherent_wide_k6_calls_for_test(operation);
                const auto rows=Exl3CudaLinearWorkspace::
                    coherent_wide_k6_rows_for_test(operation);
                const auto split10_calls=Exl3CudaLinearWorkspace::
                    coherent_wide_k6_split10_calls_for_test(operation);
                std::cout<<"TARGETREQUEST_COHERENT_WIDE_K6 op="
                         <<coherent_wide_names[operation]
                         <<" enabled="<<coherent_wide_k6
                         <<" calls="<<calls<<" rows="<<rows
                         <<" split10_calls="<<split10_calls<<'\n';
                require(coherent_wide_k6 ?
                        calls>0 && rows>=calls :
                        calls==0 && rows==0,
                    "coherent wide K6 target-only dispatch mismatch");
                require(coherent_wide_split10 ?
                        split10_calls>0 && split10_calls==calls :
                        split10_calls==0,
                    "coherent wide K6 split10 target-only dispatch mismatch");
            }
            return 0;
        }
        if (mode == "contextisolation") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            const std::string out_path = env("NINFER_E5A4_OUT");
            require(!prompt_file.empty() && !out_path.empty(),
                    "c2isolation needs PROMPT_FILE and OUT");
            require(env("NINFER_E5A4_TTFT_OUT").empty() &&
                        env("NINFER_E5A4_HANDOFF_PROFILE_PREFIX").empty() &&
                        env("NINFER_EXL3_TARGET_PROJECTION_TIMING").empty() &&
                        env("NINFER_E5A4_RING_CHECK_OUT").empty(),
                    "c2isolation forbids acceptance profiling outputs");
            run_context_isolation_qualification(*target, load_ids(prompt_file), out_path);
            return 0;
        }
        if (mode == "graphcurrentstate") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            const std::string out_path = env("NINFER_E5A4_OUT");
            require(!prompt_file.empty() && !out_path.empty(),
                    "graphcurrentstate needs PROMPT_FILE and OUT");
            require(env("NINFER_E5A4_TTFT_OUT").empty() &&
                        env("NINFER_E5A4_HANDOFF_PROFILE_PREFIX").empty() &&
                        env("NINFER_EXL3_TARGET_PROJECTION_TIMING").empty() &&
                        env("NINFER_E5A4_RING_CHECK_OUT").empty(),
                    "graphcurrentstate forbids acceptance profiling outputs");
            run_graph_current_state(*target, load_ids(prompt_file), out_path, max_ctx);
            return 0;
        }
        if (mode == "graphcurrentreplay") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            const std::string out_path = env("NINFER_E5A4_OUT");
            require(!prompt_file.empty() && !out_path.empty(),
                    "graphcurrentreplay needs PROMPT_FILE and OUT");
            require(env("NINFER_E5A4_TTFT_OUT").empty() &&
                        env("NINFER_E5A4_HANDOFF_PROFILE_PREFIX").empty() &&
                        env("NINFER_EXL3_TARGET_PROJECTION_TIMING").empty() &&
                        env("NINFER_E5A4_RING_CHECK_OUT").empty(),
                    "graphcurrentreplay forbids acceptance profiling outputs");
            run_graph_current_replay(*target, load_ids(prompt_file), out_path, max_ctx);
            return 0;
        }
        if (mode == "graphboundaryreplay") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            const std::string out_path = env("NINFER_E5A4_OUT");
            require(!prompt_file.empty() && !out_path.empty(),
                    "graphboundaryreplay needs PROMPT_FILE and OUT");
            require(env("NINFER_E5A4_TTFT_OUT").empty() &&
                        env("NINFER_E5A4_HANDOFF_PROFILE_PREFIX").empty() &&
                        env("NINFER_EXL3_TARGET_PROJECTION_TIMING").empty() &&
                        env("NINFER_E5A4_RING_CHECK_OUT").empty(),
                    "graphboundaryreplay forbids acceptance profiling outputs");
            run_graph_boundary_replay(*target, load_ids(prompt_file), out_path, max_ctx);
            return 0;
        }
        if (mode == "ttfttarget") {
            require(ttft && oscar_requested, "target-only TTFT requires trace and canonical OSCAR");
            const int count = env_int("NINFER_E5A4_CONTEXTS", 0);
            auto prefix = load_ids(env("NINFER_E5A4_PROMPT_FILE"));
            require(count >= 32 && count <= max_ctx && count <= static_cast<int>(prefix.size()), "target-only TTFT prefix extent");
            prefix.resize(count);
            require(prefix.back() != kMaskToken, "target-only TTFT prefix cannot end in mask");
            TtftTrace::scope("target_only", count);
            TtftTrace::mark("context_begin");
            auto ctx = target->create_context(false);
            TtftTrace::mark("context_complete", ctx->position(), -1, true);
            TtftTrace::mark("oscar_attach_begin");
            require(ctx->try_enable_oscar_from_environment(), "target-only TTFT OSCAR attach");
            TtftTrace::mark("oscar_attach_complete", ctx->position(), -1, true);
            const double ingestion_us = ingest_prefix(*ctx, prefix, CommitSink{});
            TtftTrace::mark("seed_begin", ctx->position());
            const auto seed = sample_target(*ctx);
            TtftTrace::mark("first_token", ctx->position(), seed);
            TtftTrace::mark("graph_eager_no_setup", ctx->position());
            std::cout << "TTFTTARGET PASS context=" << count << " position=" << ctx->position()
                      << " seed=" << seed << " ingestion_us=" << ingestion_us << "\nTTFT_TARGET_DONE\n";
            return 0;
        }
        if (mode == "verifywidth") {
            run_verification_width_profile(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")), env("NINFER_E5A4_OUT"));
            return 0;
        }
        if (mode == "numericattentiontiledt71") {
            run_numeric_attention_tiled_t71_oracle();
            return 0;
        }
        if (mode == "numericattentionrows2t73") {
            run_numeric_attention_tiled_t71_oracle(false, true);
            return 0;
        }
        if (mode == "numericattentionsplitkt71b") {
            run_numeric_attention_splitk_t71b_oracle();
            return 0;
        }
        if (mode == "exactattentionnativecap") {
            run_exact_attention_extended_cap_oracle(32768);
            return 0;
        }
        if (mode == "exactattentioncap64") {
            run_exact_attention_extended_cap_oracle(65536);
            return 0;
        }
        if (mode == "exactattentioncap128") {
            run_exact_attention_extended_cap_oracle(131072);
            return 0;
        }
        if (mode == "extendedcontextcontract") {
            run_extended_context_storage_contract(*target,65536);
            return 0;
        }
        if (mode == "extendedcontextcontract128") {
            run_extended_context_storage_contract(*target,131072);
            return 0;
        }
        if (mode == "branchref") {
            run_branch_reference_qualification(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "coalescedcontext") {
            run_coalesced_context_parity(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "exacthost") {
            run_exact_host_state_qualification(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "warmtier") {
            run_warm_tier_qualification(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "turboanglel1") {
            run_turboangle_l1_qualification(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "hierarchicalsync") {
            run_hierarchical_sync_qualification(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "hierarchical4k") {
            run_hierarchical_4k_profile(*target);
            return 0;
        }
        if (mode == "hierarchicalasync") {
            run_hierarchical_async_qualification(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "hierarchicaladaptive") {
            run_hierarchical_adaptive_qualification(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "residentroot") {
            run_exact_resident_root_qualification(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "residentroott3") {
            run_exact_resident_root_t3(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),
                load_ids(env("NINFER_TEST_STRUCTURED_FILE")),
                load_ids(env("NINFER_TEST_HELDOUT_FILE")));
            return 0;
        }
        if (mode == "hostkvbatchsync") {
            run_exact_host_kv_batch_sync_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "hostkvbatchsynct3") {
            run_exact_host_kv_batch_sync_t3(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),
                load_ids(env("NINFER_TEST_STRUCTURED_FILE")),
                load_ids(env("NINFER_TEST_HELDOUT_FILE")));
            return 0;
        }
        if (mode == "hostkvbatchcopy") {
            run_exact_host_kv_batch_copy_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "hostkvbatchcopyt3") {
            run_exact_host_kv_batch_copy_t3(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),
                load_ids(env("NINFER_TEST_STRUCTURED_FILE")),
                load_ids(env("NINFER_TEST_HELDOUT_FILE")));
            return 0;
        }
        if (mode == "hostkvpinnedchunks") {
            run_exact_host_kv_pinned_chunks_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "hostkvpinnedchunksdefault") {
            run_exact_host_kv_pinned_chunks_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),load_ids(env("NINFER_TEST_PROSE_FILE")),true);
            return 0;
        }
        if (mode == "hostkvpinnedchunkst3") {
            run_exact_host_kv_pinned_chunks_t3(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),load_ids(env("NINFER_TEST_STRUCTURED_FILE")),
                load_ids(env("NINFER_TEST_HELDOUT_FILE")));
            return 0;
        }
        if (mode == "hostkvpinnedd2h") {
            run_exact_host_kv_pinned_d2h_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "hostkvpinnedd2hdefault") {
            run_exact_host_kv_pinned_d2h_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),true);
            return 0;
        }
        if (mode == "hostkvpinnedd2ht3") {
            run_exact_host_kv_pinned_d2h_t3(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),
                load_ids(env("NINFER_TEST_STRUCTURED_FILE")),
                load_ids(env("NINFER_TEST_HELDOUT_FILE")));
            return 0;
        }
        if (mode == "hostkvdeferredscatter") {
            run_exact_host_kv_deferred_scatter_screen(
                *target, load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "hostkvbankedd2h") {
            run_exact_host_kv_banked_d2h_screen(
                *target, load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "eagermlpgateup") {
            run_exact_host_kv_eager_mlp_gateup_concurrency(
                *target, load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "smallmfusedgateup") {
            run_exact_host_kv_small_m_fused_gate_up_transform(
                *target, load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "prefillk8kvasynca") {
            run_prefill_k8_kv_async_a_qualification(*target);
            return 0;
        }
        if (mode == "hostkvuniquetailreuse") {
            run_exact_host_kv_unique_tail_reuse_screen(
                *target, load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "hostkvuniquetailreuset1") {
            run_exact_host_kv_unique_tail_reuse_t1(
                *target, load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "hostkvdeferredscattert3") {
            run_exact_host_kv_pinned_d2h_t3(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),
                load_ids(env("NINFER_TEST_STRUCTURED_FILE")),
                load_ids(env("NINFER_TEST_HELDOUT_FILE")),true);
            return 0;
        }
        if (mode == "exactattentiongqasix") {
            run_exact_attention_gqa_six_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "exactattentiongqasixscores") {
            run_exact_attention_gqa_six_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),true);
            return 0;
        }
        if (mode == "exactattentiongqasixscoresdefault") {
            run_exact_attention_gqa_six_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),true,true);
            return 0;
        }
        if (mode == "exactattentiongqasixvaluessharded") {
            run_exact_attention_gqa_six_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),true,false,true);
            return 0;
        }
        if (mode == "exactattentiongqasixscorest3") {
            run_exact_attention_gqa_six_scores_t3(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),load_ids(env("NINFER_TEST_PROSE_FILE")),
                load_ids(env("NINFER_TEST_STRUCTURED_FILE")),load_ids(env("NINFER_TEST_HELDOUT_FILE")));
            return 0;
        }
        if (mode == "exacthostprefillprofile") {
            const auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));
            require(max_ctx==4352 && ids.size()>=4096,
                "exact host prefill profile requires 4K fixture and 4352 capacity");
            require(env("NINFER_EXL3_EXACT_HOST_KV_BATCH_SYNC").empty() &&
                env("NINFER_EXL3_EXACT_HOST_KV_BATCH_COPY").empty(),
                "exact host prefill profile requires qualified implicit transfer defaults");
            _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
            _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
            auto context=target->create_context(true);context->prepare_continuation(8);
            cuda_check(cudaProfilerStart(),"exact host prefill profile begin");
            const auto started=std::chrono::steady_clock::now();
            const auto request=ninfer::exl3::Exl3VeriCacheRequest::initialize(
                *context,std::span<const std::int64_t>(ids.data(),4096),1024);
            const double wall_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
            cuda_check(cudaDeviceSynchronize(),"exact host prefill profile complete");
            cuda_check(cudaProfilerStop(),"exact host prefill profile end");
            const auto stats=context->host_kv_stats();
            require(request->state()->position()==4096 && stats.completed_rows==4096 &&
                stats.copy_submissions>0 && stats.copy_submissions<stats.transfer_calls,
                "exact host prefill profile completion/accounting");
            std::cout << "EXACT_HOST_PREFILL_PROFILE_DONE prefix=4096 wall_ms=" << wall_ms
                << " h2d_bytes=" << stats.h2d_bytes << " d2h_bytes=" << stats.d2h_bytes
                << " logical_transfers=" << stats.transfer_calls
                << " copy_submissions=" << stats.copy_submissions << std::endl;
            return 0;
        }
        if (mode == "hostkvgdngraphoracle" ||
            mode == "hostkvmlptailgraphoracle" ||
            mode == "hostkvrecurrenttraceoracle") {
            const bool mlp_tail_mode=mode=="hostkvmlptailgraphoracle";
            const bool recurrent_trace_mode=mode=="hostkvrecurrenttraceoracle";
            const auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));
            require(max_ctx==4352 && ids.size()>=4096,
                "HostKV GDN graph oracle requires 4K fixture and 4352 capacity");
            _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
            _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
            _putenv_s("NINFER_EXL3_HOST_KV_FULL_LAYER_GRAPHS","0");
            _putenv_s("NINFER_EXL3_HOST_KV_MLP_TAIL_GRAPHS","0");
            _putenv_s("NINFER_EXL3_HOST_KV_TRANSACTION_RECURRENT_TRACE","0");
            _putenv_s("NINFER_EXL3_PREFILL_PROJECTION_GRAPHS","0");
            _putenv_s("NINFER_EXL3_PREFILL_PROJECTION_CHAIN_GRAPHS","0");
            _putenv_s("NINFER_EXL3_HOST_KV_GDN_SEGMENT_GRAPHS","0");
            auto eager=target->create_context(true);
            if(recurrent_trace_mode)
                _putenv_s("NINFER_EXL3_HOST_KV_TRANSACTION_RECURRENT_TRACE","1");
            else if(mlp_tail_mode)
                _putenv_s("NINFER_EXL3_HOST_KV_MLP_TAIL_GRAPHS","1");
            else
                _putenv_s("NINFER_EXL3_HOST_KV_GDN_SEGMENT_GRAPHS","1");
            auto graphed=target->create_context(true);
            constexpr std::string_view request_contract=
                "hostkvgdngraphoracle-v1";
            eager->bind_request_compatibility(std::string(request_contract));
            graphed->bind_request_compatibility(std::string(request_contract));
            eager->prepare_transaction();eager->prepare_continuation(8);
            graphed->prepare_transaction();graphed->prepare_continuation(8);
            const bool cross_request=env("NINFER_E5A4_GDN_GRAPH_CROSS_REQUEST")=="1";
            if(cross_request) {
                const auto warm_prefix=std::span<const std::int64_t>(ids.data(),128);
                const auto eager_warm=ninfer::exl3::Exl3VeriCacheRequest::initialize(
                    *eager,warm_prefix,128);
                const auto graphed_warm=ninfer::exl3::Exl3VeriCacheRequest::initialize(
                    *graphed,warm_prefix,128);
                require(eager_warm->state()->same_payload(*graphed_warm->state()),
                    "HostKV GDN graph oracle warm root mismatch");
                const auto warm_pending=sample_target(*eager);
                require(warm_pending==sample_target(*graphed),
                    "HostKV GDN graph oracle warm pending mismatch");
                std::vector<std::int64_t> warm_attempted{warm_pending};
                eager->begin_transaction();eager->decode(warm_pending);
                for(int row=1;row<8;++row) {
                    warm_attempted.push_back(sample_target(*eager));
                    if(row+1<8)eager->decode(warm_attempted.back());
                }
                eager->rollback_transaction();
                warm_attempted[3]=(warm_attempted[3]+1)%kVocab;
                const auto eager_warm_result=verify_exl3_outer_checkpointed_reference(
                    *eager,*eager_warm->state(),warm_attempted,true,{},nullptr,true);
                const auto graphed_warm_result=verify_exl3_outer_checkpointed_reference(
                    *graphed,*graphed_warm->state(),warm_attempted,true,{},nullptr,true);
                require(eager_warm_result.committed_tokens==
                        graphed_warm_result.committed_tokens &&
                        eager_warm_result.committed_state->same_payload(
                            *graphed_warm_result.committed_state),
                    "HostKV GDN graph oracle warm verification mismatch");
                eager->reset_for_request(request_contract);
                graphed->reset_for_request(request_contract);
            }
            const auto prefix=std::span<const std::int64_t>(ids.data(),4096);
            const auto eager_root=ninfer::exl3::Exl3VeriCacheRequest::initialize(
                *eager,prefix,1024);
            const auto graphed_root=ninfer::exl3::Exl3VeriCacheRequest::initialize(
                *graphed,prefix,1024);
            require(eager_root->state()->same_payload(*graphed_root->state()) &&
                    eager->position()==4096 && graphed->position()==4096,
                "HostKV GDN graph oracle base state mismatch");
            eager->reset_for_request(request_contract);
            graphed->reset_for_request(request_contract);
            eager->restore_exact_host_state(*eager_root->state());
            graphed->restore_exact_host_state(*graphed_root->state());
            require(eager->position()==4096 && graphed->position()==4096,
                "HostKV GDN graph oracle request restore position mismatch");
            const auto pending=sample_target(*eager);
            require(pending==sample_target(*graphed),
                "HostKV GDN graph oracle base pending mismatch");
            std::vector<std::int64_t> attempted{pending};attempted.reserve(8);
            eager->begin_transaction();
            eager->decode(pending);
            for(int row=1;row<8;++row){
                attempted.push_back(sample_target(*eager));
                if(row+1<8)eager->decode(attempted.back());
            }
            eager->rollback_transaction();
            if(const auto fixed=env("NINFER_E5A4_GDN_GRAPH_ATTEMPTED");!fixed.empty()) {
                attempted.clear();
                std::stringstream stream(fixed);
                std::string field;
                while(std::getline(stream,field,',')) {
                    require(!field.empty(),
                        "HostKV GDN graph oracle empty fixed proposal field");
                    attempted.push_back(std::stoll(field));
                }
                require(attempted.size()==8 && std::all_of(
                    attempted.begin(),attempted.end(),[](std::int64_t token) {
                        return token>=0 && token<kVocab;
                    }),"HostKV GDN graph oracle fixed proposal contract");
            }
            require(eager->position()==4096 && sample_target(*eager)==pending,
                "HostKV GDN graph oracle proposal probe rollback");
            eager->begin_transaction();graphed->begin_transaction();
            eager->continue_rows(attempted);graphed->continue_rows(attempted);
            const auto eager_logits=eager->continuation_logits_bits_host();
            const auto graphed_logits=graphed->continuation_logits_bits_host();
            const auto first_difference=[](const auto& a,const auto& b) {
                const auto count=std::min(a.size(),b.size());
                for(std::size_t index=0;index<count;++index)
                    if(a[index]!=b[index])return static_cast<long long>(index);
                return a.size()==b.size()?-1LL:static_cast<long long>(count);
            };
            const auto logits_difference=first_difference(eager_logits,graphed_logits);
            int first_state_layer=-1,first_conv_layer=-1;
            long long first_state_element=-1,first_conv_element=-1;
            for(int layer=0;layer<64;++layer) {
                if(layer%4==3)continue;
                const auto eager_state=eager->gdn_state_host(layer);
                const auto graph_state=graphed->gdn_state_host(layer);
                const auto eager_conv=eager->gdn_physical_conv_host(layer);
                const auto graph_conv=graphed->gdn_physical_conv_host(layer);
                if(first_state_layer<0 && eager_state!=graph_state) {
                    first_state_layer=layer;
                    first_state_element=first_difference(eager_state,graph_state);
                }
                if(first_conv_layer<0 && eager_conv!=graph_conv) {
                    first_conv_layer=layer;
                    first_conv_element=first_difference(eager_conv,graph_conv);
                }
                if(first_state_layer>=0 && first_conv_layer>=0)break;
            }
            const auto stats=graphed->host_kv_gdn_segment_graph_stats();
            require(((mlp_tail_mode || recurrent_trace_mode) &&
                     stats.captures==0 && stats.replays==0) ||
                    (!mlp_tail_mode && !recurrent_trace_mode && stats.captures==128 &&
                      stats.replays==(cross_request?48u:16u)),
                "HostKV GDN graph oracle route not exercised");
            const auto mlp_tail_stats=graphed->host_kv_mlp_tail_graph_stats();
            require(!mlp_tail_mode ||
                    (mlp_tail_stats.captures==128 &&
                     mlp_tail_stats.replays==(cross_request?48u:16u)),
                "HostKV MLP-tail graph oracle route not exercised");
            eager->rollback_transaction();graphed->rollback_transaction();
            auto rejected_attempted=attempted;
            if(env("NINFER_E5A4_GDN_GRAPH_ATTEMPTED").empty())
                rejected_attempted[3]=(rejected_attempted[3]+1)%kVocab;
            const auto eager_outer=verify_exl3_outer_checkpointed_reference(
                *eager,*eager_root->state(),rejected_attempted,true,{},nullptr,true);
            const auto graphed_outer=verify_exl3_outer_checkpointed_reference(
                *graphed,*graphed_root->state(),rejected_attempted,true,{},nullptr,true);
            const bool outer_tokens_equal=
                eager_outer.committed_tokens==graphed_outer.committed_tokens;
            const bool outer_state_equal=eager_outer.committed_state->same_payload(
                *graphed_outer.committed_state);
            bool outer_taps_equal=true;
            for(std::size_t tap=0;tap<eager_outer.committed_taps.size();++tap)
                outer_taps_equal&=eager_outer.committed_taps[tap]==
                    graphed_outer.committed_taps[tap];
            eager->restore_exact_host_state(*eager_root->state());
            graphed->restore_exact_host_state(*graphed_root->state());
            std::array<int,3> retained_state_layer{},retained_conv_layer{};
            std::array<long long,3> retained_state_element{},retained_conv_element{};
            std::array<int,3> retained_pending_equal{};
            retained_state_layer.fill(-1);retained_conv_layer.fill(-1);
            retained_state_element.fill(-1);retained_conv_element.fill(-1);
            long long second_logits_difference=-1;
            int second_state_layer=-1,second_conv_layer=-1;
            long long second_state_element=-1,second_conv_element=-1;
            long long correction_logits_difference=-1;
            int correction_state_layer=-1,correction_conv_layer=-1;
            long long correction_state_element=-1,correction_conv_element=-1;
            constexpr std::array<int,3> retained_rows{1,4,7};
            for(std::size_t probe=0;probe<retained_rows.size();++probe) {
                eager->begin_transaction();graphed->begin_transaction();
                eager->continue_rows(attempted);graphed->continue_rows(attempted);
                eager->retain_transaction_prefix(retained_rows[probe]);
                graphed->retain_transaction_prefix(retained_rows[probe]);
                require(eager->position()==4096+retained_rows[probe] &&
                        graphed->position()==eager->position(),
                    "HostKV GDN graph oracle retained position mismatch");
                retained_pending_equal[probe]=
                    sample_target(*eager)==sample_target(*graphed)?1:0;
                for(int layer=0;layer<64;++layer) {
                    if(layer%4==3)continue;
                    const auto eager_state=eager->gdn_state_host(layer);
                    const auto graph_state=graphed->gdn_state_host(layer);
                    const auto eager_conv=eager->gdn_physical_conv_host(layer);
                    const auto graph_conv=graphed->gdn_physical_conv_host(layer);
                    if(retained_state_layer[probe]<0 && eager_state!=graph_state) {
                        retained_state_layer[probe]=layer;
                        retained_state_element[probe]=
                            first_difference(eager_state,graph_state);
                    }
                    if(retained_conv_layer[probe]<0 && eager_conv!=graph_conv) {
                        retained_conv_layer[probe]=layer;
                        retained_conv_element[probe]=
                            first_difference(eager_conv,graph_conv);
                    }
                    if(retained_state_layer[probe]>=0 &&
                       retained_conv_layer[probe]>=0)break;
                }
                if(probe+1<retained_rows.size()) {
                    eager->rollback_transaction();graphed->rollback_transaction();
                    require(eager->position()==4096 && graphed->position()==4096 &&
                            sample_target(*eager)==pending &&
                            sample_target(*graphed)==pending,
                        "HostKV GDN graph oracle retained probe rollback");
                } else {
                    const auto correction=sample_target(*eager);
                    require(correction==sample_target(*graphed),
                        "HostKV GDN graph oracle correction token mismatch");
                    eager->decode(correction);graphed->decode(correction);
                    correction_logits_difference=first_difference(
                        eager->logits_host(),graphed->logits_host());
                    for(int layer=0;layer<64;++layer) {
                        if(layer%4==3)continue;
                        const auto eager_state=eager->gdn_state_host(layer);
                        const auto graph_state=graphed->gdn_state_host(layer);
                        const auto eager_conv=eager->gdn_physical_conv_host(layer);
                        const auto graph_conv=graphed->gdn_physical_conv_host(layer);
                        if(correction_state_layer<0 && eager_state!=graph_state) {
                            correction_state_layer=layer;
                            correction_state_element=first_difference(
                                eager_state,graph_state);
                        }
                        if(correction_conv_layer<0 && eager_conv!=graph_conv) {
                            correction_conv_layer=layer;
                            correction_conv_element=first_difference(eager_conv,graph_conv);
                        }
                        if(correction_state_layer>=0 && correction_conv_layer>=0)break;
                    }
                    eager->commit_transaction();graphed->commit_transaction();
                    const auto second_pending=sample_target(*eager);
                    require(second_pending==sample_target(*graphed),
                        "HostKV GDN graph oracle second-round pending mismatch");
                    std::vector<std::int64_t> second_attempted{second_pending};
                    second_attempted.reserve(8);
                    eager->begin_transaction();
                    eager->decode(second_pending);
                    for(int row=1;row<8;++row) {
                        second_attempted.push_back(sample_target(*eager));
                        if(row+1<8)eager->decode(second_attempted.back());
                    }
                    eager->rollback_transaction();
                    eager->begin_transaction();graphed->begin_transaction();
                    eager->continue_rows(second_attempted);
                    graphed->continue_rows(second_attempted);
                    second_logits_difference=first_difference(
                        eager->continuation_logits_bits_host(),
                        graphed->continuation_logits_bits_host());
                    for(int layer=0;layer<64;++layer) {
                        if(layer%4==3)continue;
                        const auto eager_state=eager->gdn_state_host(layer);
                        const auto graph_state=graphed->gdn_state_host(layer);
                        const auto eager_conv=eager->gdn_physical_conv_host(layer);
                        const auto graph_conv=graphed->gdn_physical_conv_host(layer);
                        if(second_state_layer<0 && eager_state!=graph_state) {
                            second_state_layer=layer;
                            second_state_element=first_difference(
                                eager_state,graph_state);
                        }
                        if(second_conv_layer<0 && eager_conv!=graph_conv) {
                            second_conv_layer=layer;
                            second_conv_element=first_difference(eager_conv,graph_conv);
                        }
                        if(second_state_layer>=0 && second_conv_layer>=0)break;
                    }
                    eager->rollback_transaction();graphed->rollback_transaction();
                }
            }
            const auto final_stats=graphed->host_kv_gdn_segment_graph_stats();
            require(((mlp_tail_mode || recurrent_trace_mode) &&
                     final_stats.captures==0 &&
                      final_stats.replays==0) ||
                    (!mlp_tail_mode && !recurrent_trace_mode &&
                     final_stats.captures==128 &&
                      final_stats.replays==(cross_request?160u:128u)),
                "HostKV GDN graph retained probes not exercised");
            const auto final_mlp_tail_stats=
                graphed->host_kv_mlp_tail_graph_stats();
            require(!mlp_tail_mode ||
                    (final_mlp_tail_stats.captures==128 &&
                     final_mlp_tail_stats.replays==
                        (cross_request?160u:128u)),
                "HostKV MLP-tail graph retained probes not exercised");
            const auto recurrent_trace_stats=graphed->host_kv_stats();
            require(!recurrent_trace_mode ||
                    (recurrent_trace_stats.transaction_recurrent_trace_alias_layers>0 &&
                     recurrent_trace_stats.transaction_recurrent_trace_copy_bytes_saved>0),
                "HostKV recurrent trace oracle route not exercised");
            std::cout<<(recurrent_trace_mode?
                "HOSTKV_RECURRENT_TRACE_ORACLE PASS logits_equal=":
                (mlp_tail_mode?
                 "HOSTKV_MLP_TAIL_GRAPH_ORACLE PASS logits_equal=":
                 "HOSTKV_GDN_GRAPH_ORACLE PASS logits_equal="))
                <<(logits_difference<0?1:0)
                <<" first_logit_element="<<logits_difference
                <<" first_logit_row="<<(logits_difference<0?-1:
                    logits_difference/static_cast<long long>(kVocab))
                <<" first_state_layer="<<first_state_layer
                <<" first_state_element="<<first_state_element
                <<" first_conv_layer="<<first_conv_layer
                <<" first_conv_element="<<first_conv_element
                <<" outer_tokens_equal="<<(outer_tokens_equal?1:0)
                <<" outer_state_equal="<<(outer_state_equal?1:0)
                <<" outer_taps_equal="<<(outer_taps_equal?1:0)
                <<" outer_eager_accepted="<<eager_outer.accepted
                <<" outer_graph_accepted="<<graphed_outer.accepted
                <<" outer_eager_replay="<<eager_outer.replay_rows
                <<" outer_graph_replay="<<graphed_outer.replay_rows
                <<" retained1_pending_equal="<<retained_pending_equal[0]
                <<" retained1_state_layer="<<retained_state_layer[0]
                <<" retained1_state_element="<<retained_state_element[0]
                <<" retained1_conv_layer="<<retained_conv_layer[0]
                <<" retained1_conv_element="<<retained_conv_element[0]
                <<" retained4_pending_equal="<<retained_pending_equal[1]
                <<" retained4_state_layer="<<retained_state_layer[1]
                <<" retained4_state_element="<<retained_state_element[1]
                <<" retained4_conv_layer="<<retained_conv_layer[1]
                <<" retained4_conv_element="<<retained_conv_element[1]
                <<" retained7_pending_equal="<<retained_pending_equal[2]
                <<" retained7_state_layer="<<retained_state_layer[2]
                <<" retained7_state_element="<<retained_state_element[2]
                <<" retained7_conv_layer="<<retained_conv_layer[2]
                <<" retained7_conv_element="<<retained_conv_element[2]
                <<" correction_logits_equal="<<(correction_logits_difference<0?1:0)
                <<" correction_logit_element="<<correction_logits_difference
                <<" correction_state_layer="<<correction_state_layer
                <<" correction_state_element="<<correction_state_element
                <<" correction_conv_layer="<<correction_conv_layer
                <<" correction_conv_element="<<correction_conv_element
                <<" second_logits_equal="<<(second_logits_difference<0?1:0)
                <<" second_logit_element="<<second_logits_difference
                <<" second_logit_row="<<(second_logits_difference<0?-1:
                    second_logits_difference/static_cast<long long>(kVocab))
                <<" second_state_layer="<<second_state_layer
                <<" second_state_element="<<second_state_element
                <<" second_conv_layer="<<second_conv_layer
                <<" second_conv_element="<<second_conv_element
                <<" captures="<<final_stats.captures
                <<" replays="<<final_stats.replays
                <<" mlp_tail_captures="<<final_mlp_tail_stats.captures
                <<" mlp_tail_replays="<<final_mlp_tail_stats.replays
                <<" recurrent_trace_alias_layers="
                    <<recurrent_trace_stats.transaction_recurrent_trace_alias_layers
                <<" recurrent_trace_copy_bytes_saved="
                    <<recurrent_trace_stats.transaction_recurrent_trace_copy_bytes_saved
                <<" cross_request="<<(cross_request?1:0)
                <<std::endl;
            return 0;
        }
        if (mode == "exactattnqshared") {
            run_exact_attention_q_shared_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "exactattnqsharedt3") {
            run_exact_attention_q_shared_t3(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),load_ids(env("NINFER_TEST_PROSE_FILE")),
                load_ids(env("NINFER_TEST_STRUCTURED_FILE")),load_ids(env("NINFER_TEST_HELDOUT_FILE")));
            return 0;
        }
        if (mode == "exactattnkhalf2") {
            run_exact_attention_k_half2_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "exactattnkhalf2t3") {
            run_exact_attention_k_half2_t3(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),load_ids(env("NINFER_TEST_STRUCTURED_FILE")),
                load_ids(env("NINFER_TEST_HELDOUT_FILE")));
            return 0;
        }
        if (mode == "exactattnvhalf2") {
            run_exact_attention_v_half2_screen(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "exactattnvhalf2t3") {
            run_exact_attention_v_half2_t3(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),load_ids(env("NINFER_TEST_STRUCTURED_FILE")),
                load_ids(env("NINFER_TEST_HELDOUT_FILE")));
            return 0;
        }
        if (mode == "exactattngqapair") {
            run_exact_attention_gqa_pair_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "exactattngqapairt3") {
            run_exact_attention_gqa_pair_t3(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),load_ids(env("NINFER_TEST_STRUCTURED_FILE")),
                load_ids(env("NINFER_TEST_HELDOUT_FILE")));
            return 0;
        }
        if (mode == "exactattngqatriple") {
            run_exact_attention_gqa_triple_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "exactattngqatriplet3") {
            run_exact_attention_gqa_triple_t3(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),load_ids(env("NINFER_TEST_STRUCTURED_FILE")),
                load_ids(env("NINFER_TEST_HELDOUT_FILE")));
            return 0;
        }
        if (mode == "exactattngqatriplevalues128") {
            run_exact_attention_gqa_triple_values128_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "exactattngqatriplevalues128t3") {
            run_exact_attention_gqa_triple_values128_t3(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),load_ids(env("NINFER_TEST_STRUCTURED_FILE")),
                load_ids(env("NINFER_TEST_HELDOUT_FILE")));
            return 0;
        }
        if (mode == "exactattngqatriplevalues4") {
            run_exact_attention_gqa_triple_values4_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "exactattngqatriplesoftmaxstaged") {
            run_exact_attention_gqa_triple_softmax_staged_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "exactattngqatriplesoftmaxstagedt3") {
            run_exact_attention_gqa_triple_softmax_staged_t3(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),load_ids(env("NINFER_TEST_STRUCTURED_FILE")),
                load_ids(env("NINFER_TEST_HELDOUT_FILE")));
            return 0;
        }
        if (mode == "prefillasyncall") {
            run_prefill_async_all_screen(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "prefillrowpairk7state") {
            run_rowpair_k7_prefill_state(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "prefillasyncallt3") {
            run_prefill_async_all_t3(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),load_ids(env("NINFER_TEST_STRUCTURED_FILE")),
                load_ids(env("NINFER_TEST_HELDOUT_FILE")));
            return 0;
        }
        if (mode == "prefilldirecttiles64") {
            run_prefill_direct_tiles64_screen(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "prefillpersistingl2") {
            run_prefill_persisting_l2(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "prefilldirecttiles64t3") {
            run_prefill_direct_tiles64_t3(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),load_ids(env("NINFER_TEST_STRUCTURED_FILE")),
                load_ids(env("NINFER_TEST_HELDOUT_FILE")));
            return 0;
        }
        if (mode == "prefillreduceshfl") {
            run_prefill_reduce_shfl_screen(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "prefillreduceshflt3") {
            run_prefill_reduce_shfl_t3(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),load_ids(env("NINFER_TEST_STRUCTURED_FILE")),
                load_ids(env("NINFER_TEST_HELDOUT_FILE")));
            return 0;
        }
        if (mode == "prefillbundle") {
            run_prefill_bundle_screen(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "prefillbundledefault") {
            run_prefill_bundle_screen(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),true);
            return 0;
        }
        if (mode == "prefillbundlet3") {
            run_prefill_bundle_t3(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),load_ids(env("NINFER_TEST_STRUCTURED_FILE")),
                load_ids(env("NINFER_TEST_HELDOUT_FILE")));
            return 0;
        }
        if (mode == "gdnpaircolumns") {
            run_gdn_pair_columns_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "gdnpaircolumnsdefault") {
            run_gdn_pair_columns_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),true);
            return 0;
        }
        if (mode == "gdnpaircolumnst3") {
            run_gdn_pair_columns_t3(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),
                load_ids(env("NINFER_TEST_STRUCTURED_FILE")),
                load_ids(env("NINFER_TEST_HELDOUT_FILE")));
            return 0;
        }
        if (mode == "gdnpairvectorio") {
            run_gdn_pair_vector_io_qualification(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "gdnpairvectorioperf") {
            run_gdn_pair_vector_io_route_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "gdndualtransform") {
            run_gdn_dual_input_transform_qualification(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "gdndualtransformperf") {
            run_gdn_dual_input_transform_route_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "gdnquadcolumns") {
            run_gdn_quad_columns_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "oscarfusedkv") {
            run_oscar_fused_kv_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "oscarfusedkvt3") {
            run_oscar_fused_kv_t3(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),
                load_ids(env("NINFER_TEST_STRUCTURED_FILE")),
                load_ids(env("NINFER_TEST_HELDOUT_FILE")));
            return 0;
        }
        if (mode == "exactattngqasixsoftmaxtriplevalues") {
            run_exact_attention_gqa_six_softmax_triple_values_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "exactattngqasixsoftmaxtriplepairdimensionsstate") {
            run_exact_attention_gqa_six_softmax_triple_pair_dimensions_state(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "exactattngqasixsoftmaxsixvaluessingleloadstate") {
            run_exact_attention_gqa_six_softmax_six_values_single_load_state(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "exactattngqasixsoftmaxsixvaluesscalarsingleloadstate") {
            run_exact_attention_gqa_six_softmax_six_values_scalar_single_load_state(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "exactattngqasixsoftmaxtriplekeypairstate") {
            run_exact_attention_gqa_six_softmax_triple_key_pair_pipeline_state(
                *target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "exactattngqasixsoftmaxtriplescoretilestate") {
            run_exact_attention_gqa_six_softmax_triple_score_tile_state(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "exactattngqasixsoftmaxtriplewarpbroadcaststate") {
            run_exact_attention_gqa_six_softmax_triple_warp_score_broadcast_state(
                *target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "exactattngqasixpackedtriples") {
            run_exact_attention_gqa_six_packed_triples_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "exactattngqasixextentshards") {
            run_exact_attention_gqa_six_extent_shards_screen(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "exactcontinue") {
            run_exact_continuation_qualification(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "outerbatch") {
            run_outer_batched_qualification(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "exactattentionstate") {
            run_exact_attention_model_qualification(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "exactpaged") {
            run_exact_paged_qualification(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "native64kprefixswitch") {
            run_native64k_prefix_switch_qualification(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "hoststream") {
            run_host_stream_qualification(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "compactl0") {
            run_compact_l0_qualification(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "exactwide") {
            run_exact_wide_qualification(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "hostresidency") {
            run_host_residency_qualification(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "outerstop") {
            run_outer_stop_qualification(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "exactprofile") {
            const auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));
            require(ids.size()>=324,"exactprofile fixture extent");
            auto context=target->create_context(true);
            context->prefill(std::span<const std::int64_t>(ids.data(),16));
            for(int i=16;i<321;++i) context->decode(ids[i]);
            cuda_check(cudaDeviceSynchronize(),"exactprofile prefix complete");
            cuda_check(cudaProfilerStart(),"exactprofile begin");
            for(int i=321;i<324;++i) context->decode(ids[i]);
            cuda_check(cudaDeviceSynchronize(),"exactprofile decode complete");
            cuda_check(cudaProfilerStop(),"exactprofile end");
            std::cout << "EXACT_PROFILE_DONE rows=3 prefix=321 position=" << context->position() << '\n';
            return 0;
        }
        if (handoff_profile)
            handoff_profile->sample_memory(
                "target_loaded", -1, target.get(), nullptr, nullptr);
        if (mode == "initial16ops") {
            run_initial16_qualification(*target);
            return 0;
        }
        if (mode == "prefillgateup") {
            run_prefill_gateup_qualification(*target);
            return 0;
        }
        if (mode == "prefillk6gateupwarpgroup") {
            run_prefill_k6_gateup_warpgroup_qualification(*target);
            return 0;
        }
        if (mode == "prefillk6gateupn32paircta") {
            run_prefill_k6_gateup_n32_pair_cta_qualification(*target);
            return 0;
        }
        if(mode=="prefillk6fastdecode") {
            run_prefill_k6_fast_decode_qualification(*target);
            return 0;
        }
        if(mode=="prefillk6rowpairn64") {
            run_prefill_k6_rowpair_n64_qualification(*target);
            return 0;
        }
        if(mode=="prefillk6downrowpair") {
            run_prefill_k6_down_rowpair_qualification(*target);
            return 0;
        }
        if(mode=="prefillk6downrowpairstate") {
            run_prefill_k6_down_rowpair_state(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if(mode=="prefillshape4n64") {
            run_prefill_shape4_n64_qualification(*target);
            return 0;
        }
        if(mode=="prefillreduceminbarriers") {
            run_prefill_reduce_min_barriers_qualification(*target);
            return 0;
        }
        if (mode == "prefillk7tiles64exact") {
            run_prefill_k7_tiles64_exact_splits_qualification(*target);
            return 0;
        }
        if (mode == "prefillz") {
            run_target_z_k6_qualification(*target);
            return 0;
        }
        if (mode == "prefillrowpairops") {
            run_rowpair_prefill_operators(*target);
            return 0;
        }
        if (mode == "prefillrowpairk7ops") {
            run_rowpair_k7_prefill_operators(*target);
            return 0;
        }
        if (mode == "prefillwideops") {
            run_wide_prefill_operators(*target);
            return 0;
        }
        if (mode == "prefillprojectiongraph") {
            run_prefill_projection_graph_operator(*target);
            return 0;
        }
        if (mode == "prefillprojectionchaingraph") {
            run_prefill_projection_chain_graph_operator(*target);
            return 0;
        }
        if (mode == "prefillattentionsharedscores") {
            run_prefill_attention_shared_scores_operator();
            return 0;
        }
        if (mode == "prefillattentionsharedscorestate") {
            run_prefill_attention_shared_scores_state(
                *target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "hostkvbatchmetadata") {
            run_exact_host_kv_batch_copy_screen(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")),true);return 0;
        }
        if (mode == "packedqprojection") {
            run_packed_q_projection(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));return 0;
        }
        if (mode == "targetownerretirement") {
            run_target_owner_retirement(target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));return 0;
        }
        if (mode == "crossrequestprojectionbatch") {
            run_cross_request_projection_batch_t0(
                *target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "firstdivergence") {
            run_first_divergence(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "reconstructedexact") {
            run_reconstructed_exact_projection(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "reconstructedexactk6gateup") {
            run_reconstructed_exact_k6_gate_up(
                *target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "reconstructedexactroute") {
            run_reconstructed_exact_route(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "native16") {
            run_native16_qualification(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "reconstructedexactauthority") {
            run_reconstructed_exact_authority(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "projectionreconstructgemm") {
            run_projection_reconstruct_gemm_t69(
                *target, load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "k5prefilloperator") {
            run_k5_prefill_operator(
                *target, load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "k5directbulkoperator") {
            run_k5_direct_bulk_operator(
                *target, load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "projectionreconstructroute") {
            run_projection_reconstruct_route_t69b(
                *target, load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "projectionreconstructroutek7") {
            run_projection_reconstruct_route_t69c(
                *target, load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "numericattentionsplitkroute") {
            run_numeric_attention_splitk_route_t71b(
                *target, load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "targettxn") {
            run_target_transaction(*target);
            return 0;
        }
        if (mode == "fastdevicetxn") {
            if(env("NINFER_EXL3_TEST_FAST_DEVICE_REAL_DFLASH")=="1") {
                require(!draft_path.empty(),
                    "real device DFlash probe requires pinned draft model");
                auto real_draft=Exl3Dflash2DraftModel::load(draft_path);
                const auto prompt_ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));
                if(env("NINFER_EXL3_TEST_FAST_DEVICE_WORKER_THREAD")=="1") {
                    std::exception_ptr failure;
                    std::thread worker([&] {
                        try {
                            run_fast_device_real_dflash(*target,*real_draft,prompt_ids);
                        } catch(...) {failure=std::current_exception();}
                    });
                    worker.join();
                    if(failure)std::rethrow_exception(failure);
                } else run_fast_device_real_dflash(*target,*real_draft,prompt_ids);
            } else run_fast_device_transaction(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "targetcontinue") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            require(!prompt_file.empty(), "targetcontinue needs PROMPT_FILE");
            run_target_continuation_qualification(*target, load_ids(prompt_file));
            return 0;
        }
        if (mode == "prefixretain") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            require(!prompt_file.empty(), "prefixretain needs PROMPT_FILE");
            run_prefix_retention_qualification(*target, load_ids(prompt_file));
            return 0;
        }
        if (mode == "prefixretainodd") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            require(!prompt_file.empty(), "prefixretainodd needs PROMPT_FILE");
            run_prefix_retention_odd_qualification(*target, load_ids(prompt_file));
            return 0;
        }
        if (mode == "projectionorderedc2screen") {
            const auto prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto output_path = env("NINFER_EXL3_PROJECTION_ORDERED_C2_OUT");
            require(!prompt_file.empty() && !output_path.empty(),
                    "projectionorderedc2screen needs PROMPT_FILE and output");
            require(oscar_requested,
                    "projectionorderedc2screen requires canonical OSCAR");
            run_projection_ordered_c2_screen(
                *target, load_ids(prompt_file), output_path);
            return 0;
        }
        if (mode == "projectionroutematrixscreen") {
            const auto prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto output_directory = env("NINFER_EXL3_PROJECTION_ROUTE_MATRIX_OUT");
            require(!prompt_file.empty() && !output_directory.empty(),
                    "projectionroutematrixscreen needs PROMPT_FILE and output directory");
            require(oscar_requested,
                    "projectionroutematrixscreen requires canonical OSCAR");
            run_projection_route_matrix_screen(
                *target, load_ids(prompt_file), output_directory);
            return 0;
        }
        if (mode == "gdnstageoracle") {
            const auto prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            require(!prompt_file.empty(), "gdnstageoracle needs PROMPT_FILE");
            require(oscar_requested,
                    "gdnstageoracle requires canonical OSCAR");
            run_gdn_stage_oracle(*target, load_ids(prompt_file));
            return 0;
        }
        if (mode == "targetgraphc2screen") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_EXL3_TARGET_GRAPH_C2_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "targetgraphc2screen needs code/prose prompts and output");
            require(oscar_requested,
                    "targetgraphc2screen requires canonical OSCAR");
            run_target_graph_c2_screen(
                *target, load_ids(code_file), load_ids(prose_file), output);
            return 0;
        }
        if (mode == "targetgraphc2lifecycle") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_EXL3_TARGET_GRAPH_C2_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "targetgraphc2lifecycle needs code/prose prompts and output");
            require(oscar_requested,
                    "targetgraphc2lifecycle requires canonical OSCAR");
            run_target_graph_c2_lifecycle(
                *target, load_ids(code_file), load_ids(prose_file), output);
            return 0;
        }
        if (mode == "targetgraphresetreuse") {
            run_target_graph_reset_reuse_performance(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "prefixindex") {
            run_prefix_index_qualification(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "prefixservingt72") {
            run_prefix_serving_t72_qualification(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "prefixservingt72t1") {
            run_prefix_serving_t72_t1(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "contextreuse") {
            run_context_reuse_qualification(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "contextreuset1") {
            run_context_reuse_performance(*target,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "mediastate") {
            run_media_state_gate(*target,load_ids(env("NINFER_E5A4_PROMPT_FILE")),load_ids(env("NINFER_TEST_PROSE_FILE")));return 0;
        }
        TtftTrace::mark("draft_load_begin");
        auto draft = Exl3Dflash2DraftModel::load(draft_path);
        TtftTrace::mark("draft_load_complete", -1, -1, true);
        if (mode == "mul1lookupdisc") {
            constexpr std::uint32_t target_multiplier = 2212286765u;
            const auto result = ninfer::exl3::exl3_mul1_lookup_discriminator_for_test(
                target_multiplier);
            require(result.mismatches == 0,
                    "MUL1 lookup discriminator differs from exact arithmetic");
            std::cout << std::fixed << std::setprecision(3)
                      << "MUL1_LOOKUP_DISCRIMINATOR PASS"
                      << " multiplier=" << result.multiplier
                      << " pairs=" << result.state_pairs
                      << " table_bytes=" << result.table_bytes
                      << " table_setup_us=" << result.table_setup_us
                      << " arithmetic_median_us=" << result.arithmetic_median_us
                      << " lookup_median_us=" << result.lookup_median_us
                      << " mismatches=" << result.mismatches << '\n';
            return 0;
        }
        if (mode == "triplescorelayoutdisc") {
            const auto result=
                ninfer::exl3::exl3_triple_score_layout_discriminator_for_test();
            require(result.mismatches==0,
                    "triple score-layout discriminator differs from selected consumer");
            std::cout << std::fixed << std::setprecision(3)
                      << "TRIPLE_SCORE_LAYOUT_DISCRIMINATOR PASS"
                      << " history_rows=" << result.history_rows
                      << " score_bytes=" << result.score_bytes
                      << " head_major_median_us=" << result.head_major_median_us
                      << " interleaved_median_us=" << result.interleaved_median_us
                      << " mismatches=" << result.mismatches << '\n';
            return 0;
        }
        if (mode == "softmaxvaluefusiondisc") {
            const auto result=
                ninfer::exl3::exl3_softmax_value_fusion_discriminator_for_test();
            require(result.output_mismatches==0&&result.score_mismatches==0,
                    "softmax/value fusion discriminator differs from selected stage");
            std::cout << std::fixed << std::setprecision(3)
                      << "SOFTMAX_VALUE_FUSION_DISCRIMINATOR PASS"
                      << " history_rows=" << result.history_rows
                      << " score_bytes=" << result.score_bytes
                      << " selected_median_us=" << result.selected_median_us
                      << " fused_median_us=" << result.fused_median_us
                      << " output_mismatches=" << result.output_mismatches
                      << " score_mismatches=" << result.score_mismatches << '\n';
            return 0;
        }
        if (mode == "deferrednormalizationdisc") {
            const auto result=
                ninfer::exl3::exl3_deferred_normalization_discriminator_for_test();
            require(result.output_mismatches==0&&
                    result.normalized_score_mismatches==0,
                    "deferred normalization differs from selected stage");
            std::cout << std::fixed << std::setprecision(3)
                      << "DEFERRED_NORMALIZATION_DISCRIMINATOR PASS"
                      << " history_rows=" << result.history_rows
                      << " score_bytes=" << result.score_bytes
                      << " denominator_bytes=" << result.denominator_bytes
                      << " selected_median_us=" << result.selected_median_us
                      << " deferred_median_us=" << result.deferred_median_us
                      << " output_mismatches=" << result.output_mismatches
                      << " normalized_score_mismatches="
                      << result.normalized_score_mismatches << '\n';
            return 0;
        }
        if (mode == "splitplaneprefetchdisc") {
            const auto result=
                ninfer::exl3::exl3_split_plane_prefetch_discriminator_for_test();
            require(result.mismatches==0,
                    "split-plane prefetch differs from selected reduction");
            std::cout << std::fixed << std::setprecision(3)
                      << "SPLIT_PLANE_PREFETCH_DISCRIMINATOR PASS"
                      << " rows=" << result.rows
                      << " output_features=" << result.output_features
                      << " split_count=" << result.split_count
                      << " accumulation_bytes=" << result.accumulation_bytes
                      << " selected_median_us=" << result.selected_median_us
                      << " prefetched_median_us=" << result.prefetched_median_us
                      << " mismatches=" << result.mismatches << '\n';
            return 0;
        }
        if (mode == "attentiongatefusiondisc") {
            const auto result=
                ninfer::exl3::exl3_attention_gate_fusion_discriminator_for_test();
            require(result.mismatches==0,
                    "attention gate fusion differs from selected value plus gate stage");
            std::cout << std::fixed << std::setprecision(3)
                      << "ATTENTION_GATE_FUSION_DISCRIMINATOR PASS"
                      << " history_rows=" << result.history_rows
                      << " output_bytes=" << result.output_bytes
                      << " selected_median_us=" << result.selected_median_us
                      << " fused_median_us=" << result.fused_median_us
                      << " mismatches=" << result.mismatches << '\n';
            return 0;
        }
        if (mode == "prefillk6shareddecodedisc") {
            run_prefill_k6_shared_decode_discriminator(*target);
            return 0;
        }
        if (mode == "hostkvgdnlanoracle" ||
            mode == "hostkvgdnlanecontrolleroracle") {
            using Lane=ninfer::exl3::Exl3Dflash2Execution;
            using Coordinator=ninfer::exl3::Exl3VeriCacheServingCoordinator;
            using Request=ninfer::exl3::Exl3VeriCacheRequest;
            using Cache=ninfer::exl3::Exl3VeriCacheServingPrefixCache;
            using Identity=ninfer::exl3::Exl3VeriCacheServingIdentity;
            auto retained_draft=std::shared_ptr<Exl3Dflash2DraftModel>(
                std::move(draft));
            const auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));
            require(max_ctx==4352 && ids.size()>=4096,
                "HostKV GDN lane oracle requires 4K fixture and 4352 capacity");
            _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
            _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
            _putenv_s("NINFER_EXL3_HOST_KV_FULL_LAYER_GRAPHS","0");
            _putenv_s("NINFER_EXL3_HOST_KV_MLP_TAIL_GRAPHS","0");
            _putenv_s("NINFER_EXL3_PREFILL_PROJECTION_GRAPHS","0");
            _putenv_s("NINFER_EXL3_PREFILL_PROJECTION_CHAIN_GRAPHS","0");
            _putenv_s("NINFER_EXL3_HOST_KV_GDN_SEGMENT_GRAPHS","0");
            auto eager_context=std::shared_ptr<Exl3TextContext>(
                target->create_context(true));
            _putenv_s("NINFER_EXL3_HOST_KV_GDN_SEGMENT_GRAPHS","1");
            auto graphed_context=std::shared_ptr<Exl3TextContext>(
                target->create_context(true));
            std::array<std::unique_ptr<DeviceBuffer>,5> stage;
            std::array<std::uint16_t*,5> staging{};
            for(int tap=0;tap<5;++tap) {
                stage[tap]=std::make_unique<DeviceBuffer>(16ULL*kHidden*2);
                staging[tap]=static_cast<std::uint16_t*>(stage[tap]->get());
            }
            const auto prefix=std::span<const std::int64_t>(ids.data(),4096);
            MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);
            require(GlobalMemoryStatusEx(&memory) && memory.ullAvailPhys>(8ULL<<30),
                "HostKV GDN lane oracle physical reserve");
            const Identity identity{"exl3-engine-epoch-1","SC_6.00bpw_H6_V6",
                "0997f410-c3cf9e34","ordinary-FP16-B8-greedy","text"};
            const unsigned oracle_rounds=
                mode=="hostkvgdnlanecontrolleroracle"?8:1;
            struct LaneResult {
                std::vector<std::int64_t> proposed,committed;
                std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;
                std::array<std::vector<std::uint16_t>,5> taps;
                std::shared_ptr<const Request> initial_root;
                std::size_t accepted=0,verification_rows=0,replay_rows=0;
                bool rejected=false,stopped=false,position_current=false;
                bool pending_child=false,pending_revision_distinct=false;
                bool completion_current=false,dependency_complete=false;
                std::uint64_t captures=0,replays=0;
            };
            const auto run=[&](std::shared_ptr<Exl3TextContext> context) {
                Cache cache(Cache::Policy{2,64,2ULL<<30,8ULL<<30},identity);
                Coordinator coordinator(cache,Coordinator::Policy{
                    1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
                coordinator.bind_physical_resources({},
                    ninfer::exl3::Exl3ResourceInventory::unlimited());
                coordinator.reserve_metadata_startup();
                auto lane=std::make_unique<Lane>(context,retained_draft,
                    identity.contract(),nullptr,
                    std::shared_ptr<const void>{});
                const auto initial=Request::initialize(
                    lane->context(),prefix,1024)->compact_draft(
                        *retained_draft,staging);
                coordinator.admit(initial);
                const auto lease=coordinator.acquire();
                require(lease.has_value(),"HostKV GDN lane oracle coordinator acquire");
                require(lane->acquire(*lease,&coordinator),
                    "HostKV GDN lane oracle lane acquire");
                lane->context().set_request_metadata_reservation(
                    [&](std::uint64_t bytes) {
                        if(!lane->execution_active())throw std::logic_error(
                            "HostKV GDN lane oracle metadata lane inactive");
                        const auto current=lane->lease();
                        return coordinator.reserve_snapshot_metadata(
                            &current,{},bytes);
                    });
                LaneResult result;
                result.initial_root=initial;
                result.position_current=lane->context().position()==4096 &&
                    lane->lease().root==initial;
                result.pending_child=true;
                result.pending_revision_distinct=true;
                result.completion_current=true;
                result.dependency_complete=true;
                for(unsigned round=0;round<oracle_rounds;++round) {
                    const auto parent=lane->lease().root;
                    const auto proposed=lane->propose(8);
                    result.proposed.insert(result.proposed.end(),
                        proposed.begin(),proposed.end());
                    auto pending=lane->verify_owned(proposed);
                    const auto dependency=lane->dependency_graph_for_test();
                    result.committed.insert(result.committed.end(),
                        pending->verification.committed_tokens.begin(),
                        pending->verification.committed_tokens.end());
                    result.state=pending->verification.committed_state;
                    result.taps=pending->verification.committed_taps;
                    result.accepted+=pending->verification.accepted;
                    result.verification_rows+=pending->verification.verification_rows;
                    result.replay_rows+=pending->verification.replay_rows;
                    result.rejected|=pending->verification.rejected;
                    result.stopped|=pending->verification.stopped;
                    result.pending_child&=pending->root->is_child_of(*parent) &&
                        pending->root->state()->same_payload(*result.state);
                    result.pending_revision_distinct&=
                        pending->root->revision_owner().get()!=parent->revision_owner().get();
                    result.completion_current&=pending->completion_generation!=0 &&
                        pending->verification.committed_tap_ready_generation==
                            pending->completion_generation &&
                        pending->verification.committed_tap_ready_event!=0;
                    result.dependency_complete&=
                        dependency.phase==ninfer::exl3::Exl3ExecutionDependencyGraph::Phase::idle &&
                        dependency.nodes==0 && dependency.edges==0 &&
                        !dependency.retains_owners && dependency.first_error==0;
                    if(oracle_rounds>1) {
                        const auto publication=lane->publish(coordinator,*pending);
                        require(publication.tokens==pending->verification.committed_tokens,
                            "HostKV GDN lane controller publication tokens");
                    }
                    pending.reset();
                    if(result.stopped)break;
                }
                const auto graph=lane->context().host_kv_gdn_segment_graph_stats();
                result.captures=graph.captures;result.replays=graph.replays;
                if(oracle_rounds==1)lane->abort();
                const auto retired_lease=lane->lease();
                auto retirement=lane->release();
                coordinator.cancel_retired(retired_lease,retirement);
                lane->context().set_request_metadata_reservation({});
                coordinator.close();
                require(coordinator.stats().closed,
                    "HostKV GDN lane oracle coordinator close");
                return result;
            };
            const auto eager=run(eager_context);
            const auto graphed=run(graphed_context);
            bool taps_equal=true;
            for(std::size_t tap=0;tap<eager.taps.size();++tap)
                taps_equal&=eager.taps[tap]==graphed.taps[tap];
            const bool initial_state_equal=eager.initial_root->state()->same_payload(
                *graphed.initial_root->state());
            const bool initial_tokens_equal=eager.initial_root->same_tokens(*graphed.initial_root);
            const bool initial_input_equal=eager.initial_root->same_input_identity(*graphed.initial_root);
            const bool initial_conditioning_equal=eager.initial_root->same_projected_conditioning_for_test(
                *graphed.initial_root);
            const bool proposal_equal=eager.proposed==graphed.proposed;
            const bool committed_equal=eager.committed==graphed.committed;
            const bool state_equal=eager.state->same_payload(*graphed.state);
            std::cout<<"HOSTKV_GDN_LANE_COMPARE"
                <<" initial_state_equal="<<initial_state_equal
                <<" initial_tokens_equal="<<initial_tokens_equal
                <<" initial_input_equal="<<initial_input_equal
                <<" initial_conditioning_equal="<<initial_conditioning_equal
                <<" eager_position_current="<<eager.position_current
                <<" graph_position_current="<<graphed.position_current
                <<" proposal_equal="<<proposal_equal
                <<" committed_equal="<<committed_equal
                <<" state_equal="<<state_equal
                <<" taps_equal="<<taps_equal
                <<" eager_accepted="<<eager.accepted
                <<" graph_accepted="<<graphed.accepted
                <<" eager_verification_rows="<<eager.verification_rows
                <<" graph_verification_rows="<<graphed.verification_rows
                <<" eager_replay_rows="<<eager.replay_rows
                <<" graph_replay_rows="<<graphed.replay_rows
                <<" eager_rejected="<<eager.rejected
                <<" graph_rejected="<<graphed.rejected
                <<" eager_stopped="<<eager.stopped
                <<" graph_stopped="<<graphed.stopped
                <<" eager_pending_child="<<eager.pending_child
                <<" graph_pending_child="<<graphed.pending_child
                <<" eager_pending_revision_distinct="<<eager.pending_revision_distinct
                <<" graph_pending_revision_distinct="<<graphed.pending_revision_distinct
                <<" eager_completion_current="<<eager.completion_current
                <<" graph_completion_current="<<graphed.completion_current
                <<" eager_dependency_complete="<<eager.dependency_complete
                <<" graph_dependency_complete="<<graphed.dependency_complete
                <<" eager_captures="<<eager.captures
                <<" eager_replays="<<eager.replays
                <<" graph_captures="<<graphed.captures
                <<" graph_replays="<<graphed.replays<<std::endl;
            require(initial_state_equal && initial_tokens_equal && initial_input_equal &&
                    initial_conditioning_equal &&
                    eager.position_current && graphed.position_current &&
                    proposal_equal && committed_equal && state_equal && taps_equal &&
                    eager.accepted==graphed.accepted &&
                    eager.verification_rows==graphed.verification_rows &&
                    eager.replay_rows==graphed.replay_rows &&
                    eager.rejected==graphed.rejected &&
                    eager.stopped==graphed.stopped &&
                    eager.pending_child && graphed.pending_child &&
                    eager.pending_revision_distinct &&
                    graphed.pending_revision_distinct &&
                    eager.completion_current && graphed.completion_current &&
                    eager.dependency_complete && graphed.dependency_complete &&
                    eager.captures==0 && eager.replays==0 &&
                    graphed.captures==128 &&
                    graphed.replays==16*(oracle_rounds+graphed.replay_rows),
                "HostKV GDN real lane oracle mismatch");
            std::cout<<(oracle_rounds==1?"HOSTKV_GDN_LANE_ORACLE PASS":
                    "HOSTKV_GDN_LANE_CONTROLLER_ORACLE PASS")
                <<" rounds="<<oracle_rounds<<" proposal_equal=1"
                <<" accepted="<<eager.accepted
                <<" rejected="<<(eager.rejected?1:0)
                <<" stopped="<<(eager.stopped?1:0)
                <<" committed="<<eager.committed.size()
                <<" verification_rows="<<eager.verification_rows
                <<" replay_rows="<<eager.replay_rows
                <<" state_equal=1 taps_equal=1 position_current=1"
                <<" pending_child=1 pending_revision_distinct=1"
                <<" completion_current=1 dependency_idle=1"
                <<" eager_captures="<<eager.captures
                <<" eager_replays="<<eager.replays
                <<" graph_captures="<<graphed.captures
                <<" graph_replays="<<graphed.replays<<std::endl;
            return 0;
        }
        if (mode == "prefillattentionchaingraph") {
            run_prefill_attention_chain_graph(*target,*draft,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "mediadraft") {
            run_media_draft_gate(*target,*draft,load_ids(env("NINFER_E5A4_PROMPT_FILE")),load_ids(env("NINFER_TEST_PROSE_FILE")));return 0;
        }
        if (mode == "deviceprefixownership") {
            run_device_prefix_metadata_gate(std::filesystem::path(env("NINFER_REAL_DFLASH_OUT")).parent_path()/"device-prefix-metadata.csv");
            run_draft_execution_ownership(*target,draft,load_ids(env("NINFER_E5A4_PROMPT_FILE")),load_ids(env("NINFER_TEST_PROSE_FILE")));
            require(ninfer::exl3::Exl3DevicePrefixCache::budget_snapshot()[0]==0,"device prefix physical2 retirement");return 0;
        }
        if (mode == "historydiagnostic") {
            run_history_diagnostic(*target,*draft,load_ids(env("NINFER_E5A4_PROMPT_FILE")),load_ids(env("NINFER_TEST_PROSE_FILE")));return 0;
        }
        if (mode == "draftc2workload") {
            run_draft_c2_workload(*target,*draft,load_ids(env("NINFER_E5A4_PROMPT_FILE")),load_ids(env("NINFER_TEST_PROSE_FILE")));return 0;
        }
        if (mode == "draftexecutionownership") {
            run_draft_execution_ownership(*target,draft,load_ids(env("NINFER_E5A4_PROMPT_FILE")),load_ids(env("NINFER_TEST_PROSE_FILE")));return 0;
        }
        if (mode == "realdflashscreens") {
            run_real_dflash_screens(*target,*draft,load_ids(env("NINFER_E5A4_PROMPT_FILE")),load_ids(env("NINFER_TEST_PROSE_FILE")));return 0;
        }
        if (mode == "recurrentexportauthority") {
            run_recurrent_export_authority(*target,*draft,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "realdflashexecution") {
            run_real_dflash_execution(*target,*draft,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "native16authority") {
            std::array<std::unique_ptr<DeviceBuffer>,5> stage;
            std::array<std::uint16_t*,5> pointers{};
            for(int tap=0;tap<5;++tap) {
                stage[tap]=std::make_unique<DeviceBuffer>(16ULL*kHidden*2);
                pointers[tap]=static_cast<std::uint16_t*>(stage[tap]->get());
            }
            run_native16_authority(*target,*draft,pointers,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")),load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "tokenreplay") {
            run_token_replay_qualification(*target,*draft,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "capacity") {
            run_capacity_qualification(*target,*draft);
            return 0;
        }
        if (mode == "drafthost") {
            run_draft_host_ring_qualification(*target,*draft,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "compactrequest") {
            run_compact_request_qualification(*target,*draft,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "compactperf") {
            run_compact_draft_performance(*target,*draft);
            return 0;
        }
        if (mode == "l2window") {
            run_l2_window_experiment(*target,*draft);
            return 0;
        }
        if (mode == "hierarchicall2windows") {
            run_hierarchical_l2_windows_t70(
                *target, *draft, load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "hierarchicall2windowst1") {
            run_hierarchical_l2_windows_t70_t1(
                *target, *draft, load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "qualityprobe") {
            run_quality_probe(*target,*draft);
            return 0;
        }
        if (mode == "widerequest") {
            run_wide_request_qualification(*target,*draft,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "vericachequeue") {
            run_vericache_queue_qualification(*target,*draft,load_ids(env("NINFER_E5A4_PROMPT_FILE")),
                load_ids(env("NINFER_TEST_PROSE_FILE")));
            return 0;
        }
        if (mode == "vericacherequest") {
            run_vericache_request_qualification(*target,*draft,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "outerref") {
            run_outer_reference_qualification(*target, *draft,
                load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "prefillwidestate") {
            run_wide_prefill_state_qualification(*target,*draft);
            return 0;
        }
        if (mode == "prefillchunk") {
            run_prefill_chunk_qualification(*target,*draft);
            return 0;
        }
        std::unique_ptr<RingAttentionCapture> ring_capture;
        if (!env("NINFER_E5A4_RING_CHECK_OUT").empty()) {
            require(mode == "accept" && !handoff_profile,
                    "ring capture requires unprofiled accept qualification");
            ring_capture = std::make_unique<RingAttentionCapture>(
                *draft, env("NINFER_E5A4_RING_CHECK_OUT"));
        }
        cuda_check(cudaDeviceSynchronize(), "E5A4 load sync");
        if (handoff_profile)
            handoff_profile->sample_memory(
                "draft_loaded", -1, target.get(), draft.get(), nullptr);
        std::cout << "E5A4 target_model_bytes=" << target->model_bytes()
                  << " draft_weight_bytes=" << draft->weight_bytes() << "\n";
        if (mode == "h6qual") {
            const auto out_path = env("NINFER_E5A4_OUT");
            require(!out_path.empty(), "H6 qualification requires output path");
            std::ofstream out(out_path);
            require(out.good(), "cannot open H6 qualification report");
            run_h6qualification(*target, out);
            return 0;
        }
        if (mode == "h6sweep") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            const std::string out_path = env("NINFER_E5A4_OUT");
            require(!prompt_file.empty() && !out_path.empty(), "E5A4 h6sweep needs PROMPT_FILE and OUT");
            const auto ids = load_ids(prompt_file);
            const std::vector<std::int64_t> prompt(ids.begin(), ids.begin() + 64);
            std::ofstream out(out_path);
            require(out.good(), "E5A4 cannot open h6 out");
            out << "M,dispatch,K,in_features,out_features,med_us,wall_us,per_row_us,row0_maxabs,row0_rel,allrow_maxabs,allrow_rel\n";
            run_h6sweep(*target, prompt, out);
            std::cout << "E5A4_H6SWEEP_DONE\n";
            return 0;
        }
        if (mode == "draftsmallqual") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            const std::string out_path = env("NINFER_E5A4_OUT");
            require(!prompt_file.empty() && !out_path.empty(),
                    "draftsmallqual needs PROMPT_FILE and OUT");
            const auto ids = load_ids(prompt_file);
            require(ids.size() >= 512, "draftsmallqual prompt has fewer than 512 tokens");
            require(env_int("NINFER_E5A4_BLOCK", 8) == 8,
                    "draftsmallqual requires NINFER_E5A4_BLOCK=8");
            std::vector<std::int64_t> prompt(ids.begin(), ids.begin() + 512);
            require(prompt.back() != kMaskToken,
                    "draftsmallqual ctx512 ends in the draft mask token");
            std::ofstream out(out_path);
            require(out.good(), "cannot open draftsmallqual report");
            run_draft_small_m_qualification(*target, *draft, prompt, oscar_requested, out);
            std::cout << "E5A5K5_QUALIFICATION_DONE\n";
            return 0;
        }
        if (mode == "pendingqual") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            require(!prompt_file.empty(), "pendingqual needs PROMPT_FILE");
            run_pending_qualification(*target, *draft, load_ids(prompt_file));
            return 0;
        }
        if (mode == "transactionqual") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            require(!prompt_file.empty(), "transactionqual needs PROMPT_FILE");
            run_transaction_round_qualification(*target, *draft, load_ids(prompt_file));
            return 0;
        }
        if (mode == "targetk6capture") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            const std::string output_path = env("NINFER_EXL3_TARGET_K6_CAPTURE_OUT");
            require(!prompt_file.empty() && !output_path.empty(),
                    "targetk6capture needs PROMPT_FILE and TARGET_K6_CAPTURE_OUT");
            const auto ids = load_ids(prompt_file);
            require(ids.size() >= 512,
                    "targetk6capture prompt has fewer than 512 tokens");
            require(env_int("NINFER_E5A4_BLOCK", 8) == 8,
                    "targetk6capture requires NINFER_E5A4_BLOCK=8");
            std::vector<std::int64_t> prompt(ids.begin(), ids.begin() + 512);
            require(prompt.back() != kMaskToken,
                    "targetk6capture ctx512 ends in the draft mask token");
            run_target_k6_capture(*target, *draft, prompt, output_path,
                                  oscar_requested);
            return 0;
        }
        if (mode == "transactionretainqual") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            require(!prompt_file.empty(), "transactionretainqual needs PROMPT_FILE");
            run_transaction_retention_qualification(
                *target, *draft, load_ids(prompt_file));
            return 0;
        }
        if (mode == "transactionretainoddqual") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            require(!prompt_file.empty(), "transactionretainoddqual needs PROMPT_FILE");
            run_transaction_retention_odd_qualification(
                *target, *draft, load_ids(prompt_file));
            return 0;
        }
        if (mode == "targetdownk6capture") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            const std::string output_path =
                env("NINFER_EXL3_TARGET_DOWN_K6_CAPTURE_OUT");
            require(!prompt_file.empty() && !output_path.empty(),
                    "targetdownk6capture needs PROMPT_FILE and TARGET_DOWN_K6_CAPTURE_OUT");
            require(env("NINFER_EXL3_TARGET_GATEUP_SMALL_M") == "0",
                    "targetdownk6capture requires explicit gate/up candidate flag0");
            const auto ids = load_ids(prompt_file);
            require(ids.size() >= 512,
                    "targetdownk6capture prompt has fewer than 512 tokens");
            require(env_int("NINFER_E5A4_BLOCK", 8) == 8,
                    "targetdownk6capture requires NINFER_E5A4_BLOCK=8");
            std::vector<std::int64_t> prompt(ids.begin(), ids.begin() + 512);
            require(prompt.back() != kMaskToken,
                    "targetdownk6capture ctx512 ends in the draft mask token");
            run_target_down_k6_capture(
                *target, *draft, prompt, output_path, oscar_requested);
            return 0;
        }
        if (mode == "targetok7capture") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            const std::string output_path =
                env("NINFER_EXL3_TARGET_O_K7_CAPTURE_OUT");
            require(!prompt_file.empty() && !output_path.empty(),
                    "targetok7capture needs PROMPT_FILE and TARGET_O_K7_CAPTURE_OUT");
            require(env("NINFER_EXL3_TARGET_GATEUP_SMALL_M") == "1" &&
                        env("NINFER_EXL3_TARGET_DOWN_SMALL_M") == "1",
                    "targetok7capture requires accepted gate/up and down candidates");
            const auto ids = load_ids(prompt_file);
            require(ids.size() >= 512,
                    "targetok7capture prompt has fewer than 512 tokens");
            require(env_int("NINFER_E5A4_BLOCK", 8) == 8,
                    "targetok7capture requires NINFER_E5A4_BLOCK=8");
            std::vector<std::int64_t> prompt(ids.begin(), ids.begin() + 512);
            require(prompt.back() != kMaskToken,
                    "targetok7capture ctx512 ends in the draft mask token");
            run_target_o_k7_capture(
                *target, *draft, prompt, output_path, oscar_requested);
            return 0;
        }
        if (mode == "freshdraftprefillqual") {
            run_fresh_draft_prefill_qualification(*target,*draft,load_ids(env("NINFER_E5A4_PROMPT_FILE")));
            return 0;
        }
        if (mode == "freshdraftprefillstate") {
            run_transaction_retention_qualification(*target,*draft,load_ids(env("NINFER_E5A4_PROMPT_FILE")),false,true);
            return 0;
        }
        if (mode == "targetqkvk6state") {
            run_transaction_retention_qualification(*target,*draft,load_ids(env("NINFER_E5A4_PROMPT_FILE")),true);
            return 0;
        }
        if (mode == "targetkvqual") {
            run_target_kv_qualification(*target, *draft);
            return 0;
        }
        if (mode == "targetgateupk5qual") {
            run_target_gateup_k5_qualification(*target, *draft);
            return 0;
        }
        if (mode == "targetk5smallmbatchqual") {
            run_target_k5_small_m_batch_qualification(*target, *draft);
            return 0;
        }
        if (mode == "targetk6m1simtqual") {
            run_target_k6_m1_simt_qualification(*target,*draft);
            return 0;
        }
        if (mode == "targetok6qual") {
            run_target_o_k6_qualification(*target, *draft);
            return 0;
        }
        if (mode == "targetqkvk6qual") {
            run_target_qkv_k6_qualification(*target, *draft);
            return 0;
        }
        if (mode == "targetgraphc2verifiert1") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_EXL3_TARGET_GRAPH_C2_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "targetgraphc2verifiert1 needs code/prose prompts and output");
            require(oscar_requested,
                    "targetgraphc2verifiert1 requires canonical OSCAR");
            run_target_graph_c2_verifier_t1(
                *target, *draft, draft_path, load_ids(code_file),
                load_ids(prose_file), output);
            return 0;
        }
        if (mode == "t74realc2") {
            const auto prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto output = env("NINFER_T74_OUT");
            require(!prompt_file.empty() && !output.empty(),
                    "t74realc2 needs PROMPT_FILE and T74_OUT");
            require(oscar_requested, "t74realc2 requires canonical OSCAR");
            run_t74_real_c2(*target, *draft, draft_path,
                            load_ids(prompt_file), output);
            return 0;
        }
        if (mode == "t78cachedc2") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_T78_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "t78cachedc2 needs code/prose prompts and output");
            require(oscar_requested, "t78cachedc2 requires canonical OSCAR");
            run_t78_cached_prefix_c2(*target, *draft, draft_path,
                load_ids(code_file), load_ids(prose_file), output);
            return 0;
        }
        if (mode == "t81cachedc4") {
            const auto code_file = env("NINFER_E5A4_PROMPT_FILE");
            const auto prose_file = env("NINFER_TEST_PROSE_FILE");
            const auto output = env("NINFER_T81_OUT");
            require(!code_file.empty() && !prose_file.empty() && !output.empty(),
                    "t81cachedc4 needs code/prose prompts and output");
            require(oscar_requested, "t81cachedc4 requires canonical OSCAR");
            run_t81_cached_prefix_c4_admission(
                *target, *draft, draft_path, load_ids(code_file),
                load_ids(prose_file), output);
            return 0;
        }
        if (mode == "targetgateupk7qual") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            const std::string out_path = env("NINFER_E5A4_OUT");
            require(!prompt_file.empty() && !out_path.empty(),
                    "targetgateupk7qual needs PROMPT_FILE and OUT");
            const auto ids = load_ids(prompt_file);
            require(ids.size() >= 512,
                    "targetgateupk7qual prompt has fewer than 512 tokens");
            require(env_int("NINFER_E5A4_BLOCK", 8) == 8,
                    "targetgateupk7qual requires block8");
            std::vector<std::int64_t> prompt(ids.begin(), ids.begin() + 512);
            require(prompt.back() != kMaskToken,
                    "targetgateupk7qual ctx512 ends in mask token");
            std::ofstream out(out_path);
            require(out.good(), "cannot open targetgateupk7qual report");
            run_target_gateup_k7_qualification(
                *target, *draft, prompt, oscar_requested, out);
            return 0;
        }
        if (mode == "targetqk6qual") {
            run_target_q_k6_qualification(*target, *draft);
            return 0;
        }
        if (mode == "targetk6qual") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            const std::string out_path = env("NINFER_E5A4_OUT");
            require(!prompt_file.empty() && !out_path.empty(),
                    "targetk6qual needs PROMPT_FILE and OUT");
            const auto ids = load_ids(prompt_file);
            require(ids.size() >= 512,
                    "targetk6qual prompt has fewer than 512 tokens");
            require(env_int("NINFER_E5A4_BLOCK", 8) == 8,
                    "targetk6qual requires NINFER_E5A4_BLOCK=8");
            std::vector<std::int64_t> prompt(ids.begin(), ids.begin() + 512);
            require(prompt.back() != kMaskToken,
                    "targetk6qual ctx512 ends in the draft mask token");
            std::ofstream out(out_path);
            require(out.good(), "cannot open targetk6qual report");
            run_target_k6_qualification(
                *target, *draft, prompt, oscar_requested, out);
            return 0;
        }
        if (mode == "targetdownk6qual") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            const std::string out_path = env("NINFER_E5A4_OUT");
            require(!prompt_file.empty() && !out_path.empty(),
                    "targetdownk6qual needs PROMPT_FILE and OUT");
            require(env("NINFER_EXL3_TARGET_DOWN_SMALL_M") == "0" &&
                        env("NINFER_EXL3_TARGET_GATEUP_SMALL_M") == "0",
                    "targetdownk6qual requires both candidates explicit flag0");
            const auto ids = load_ids(prompt_file);
            require(ids.size() >= 512,
                    "targetdownk6qual prompt has fewer than 512 tokens");
            require(env_int("NINFER_E5A4_BLOCK", 8) == 8,
                    "targetdownk6qual requires NINFER_E5A4_BLOCK=8");
            std::vector<std::int64_t> prompt(ids.begin(), ids.begin() + 512);
            require(prompt.back() != kMaskToken,
                    "targetdownk6qual ctx512 ends in the draft mask token");
            std::ofstream out(out_path);
            require(out.good(), "cannot open targetdownk6qual report");
            run_target_down_k6_qualification(
                *target, *draft, prompt, oscar_requested, out);
            return 0;
        }
        if (mode == "targetdownk7qual") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            const std::string out_path = env("NINFER_E5A4_OUT");
            require(!prompt_file.empty() && !out_path.empty(),
                    "targetdownk7qual needs PROMPT_FILE and OUT");
            require(env("NINFER_EXL3_TARGET_DOWN_K7_SMALL_M") == "0" &&
                        env("NINFER_EXL3_TARGET_GATEUP_SMALL_M") == "0",
                    "targetdownk7qual requires both candidates explicit flag0");
            const auto ids = load_ids(prompt_file);
            require(ids.size() >= 512,
                    "targetdownk7qual prompt has fewer than 512 tokens");
            require(env_int("NINFER_E5A4_BLOCK", 8) == 8,
                    "targetdownk7qual requires NINFER_E5A4_BLOCK=8");
            std::vector<std::int64_t> prompt(ids.begin(), ids.begin() + 512);
            require(prompt.back() != kMaskToken,
                    "targetdownk7qual ctx512 ends in the draft mask token");
            std::ofstream out(out_path);
            require(out.good(), "cannot open targetdownk7qual report");
            run_target_down_k7_qualification(
                *target, *draft, prompt, oscar_requested, out);
            return 0;
        }
        if (mode == "targetk6smallmasyncqual") {
            const std::string prompt_file=env("NINFER_E5A4_PROMPT_FILE");
            const std::string out_path=env("NINFER_E5A4_OUT");
            require(!prompt_file.empty()&&!out_path.empty(),
                "targetk6smallmasyncqual needs PROMPT_FILE and OUT");
            require(oscar_requested,
                "targetk6smallmasyncqual requires canonical OSCAR");
            const auto ids=load_ids(prompt_file);
            require(ids.size()>=512,
                "targetk6smallmasyncqual prompt has fewer than 512 tokens");
            std::vector<std::int64_t> prompt(ids.begin(),ids.begin()+512);
            require(prompt.back()!=kMaskToken,
                "targetk6smallmasyncqual ctx512 ends in the draft mask token");
            std::ofstream out(out_path);
            require(out.good(),"cannot open targetk6smallmasyncqual report");
            run_target_k6_small_m_async_qualification(*target,*draft,prompt,out);
            return 0;
        }
        if (mode == "targetm1k6qual") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            const std::string out_path = env("NINFER_E5A4_OUT");
            require(!prompt_file.empty() && !out_path.empty(),
                    "targetm1k6qual needs PROMPT_FILE and OUT");
            const auto ids = load_ids(prompt_file);
            require(ids.size() >= 512,
                    "targetm1k6qual prompt has fewer than 512 tokens");
            std::vector<std::int64_t> prompt(ids.begin(), ids.begin() + 512);
            require(prompt.back() != kMaskToken,
                    "targetm1k6qual ctx512 ends in the draft mask token");
            std::ofstream out(out_path);
            require(out.good(), "cannot open targetm1k6qual report");
            run_target_m1_k6_qualification(
                *target, *draft, prompt, oscar_requested, out);
            return 0;
        }
        if (mode == "fastsameweightsint8qual") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            const std::string out_path = env("NINFER_E5A4_OUT");
            require(!prompt_file.empty() && !out_path.empty(),
                    "fastsameweightsint8qual needs PROMPT_FILE and OUT");
            const auto ids = load_ids(prompt_file);
            require(ids.size() >= 512,
                    "fastsameweightsint8qual prompt has fewer than 512 tokens");
            std::vector<std::int64_t> prompt(ids.begin(), ids.begin() + 512);
            require(prompt.back() != kMaskToken,
                    "fastsameweightsint8qual ctx512 ends in the draft mask token");
            std::ofstream out(out_path);
            require(out.good(), "cannot open fastsameweightsint8qual report");
            run_fast_same_weights_int8_qualification(
                *target, *draft, prompt, oscar_requested, out);
            return 0;
        }
        if (mode == "m1predecodeddisc") {
            const std::string prompt_file=env("NINFER_E5A4_PROMPT_FILE");
            require(!prompt_file.empty(),
                    "m1predecodeddisc needs PROMPT_FILE");
            const auto ids=load_ids(prompt_file);
            require(ids.size()>=512,
                    "m1predecodeddisc prompt has fewer than 512 tokens");
            std::vector<std::int64_t> prompt(ids.begin(),ids.begin()+512);
            require(prompt.back()!=kMaskToken,
                    "m1predecodeddisc ctx512 ends in the draft mask token");
            run_target_m1_predecoded_discriminator(
                *target,*draft,prompt,oscar_requested);
            return 0;
        }
        if (mode == "m1gateuppairdisc") {
            const std::string prompt_file=env("NINFER_E5A4_PROMPT_FILE");
            require(!prompt_file.empty(),"m1gateuppairdisc needs PROMPT_FILE");
            const auto ids=load_ids(prompt_file);
            require(ids.size()>=512,
                "m1gateuppairdisc prompt has fewer than 512 tokens");
            std::vector<std::int64_t> prompt(ids.begin(),ids.begin()+512);
            require(prompt.back()!=kMaskToken,
                "m1gateuppairdisc ctx512 ends in the draft mask token");
            run_target_m1_gate_up_pair_discriminator(
                *target,*draft,prompt,oscar_requested);
            return 0;
        }
        if (mode == "m1fastdecodedisc") {
            const std::string prompt_file=env("NINFER_E5A4_PROMPT_FILE");
            require(!prompt_file.empty(),"m1fastdecodedisc needs PROMPT_FILE");
            const auto ids=load_ids(prompt_file);
            require(ids.size()>=512,
                "m1fastdecodedisc prompt has fewer than 512 tokens");
            std::vector<std::int64_t> prompt(ids.begin(),ids.begin()+512);
            require(prompt.back()!=kMaskToken,
                "m1fastdecodedisc ctx512 ends in the draft mask token");
            run_target_m1_fast_decode_discriminator(
                *target,*draft,prompt,oscar_requested);
            return 0;
        }
        if (mode == "targetok7qual") {
            const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
            const std::string out_path = env("NINFER_E5A4_OUT");
            require(!prompt_file.empty() && !out_path.empty(),
                    "targetok7qual needs PROMPT_FILE and OUT");
            require(env("NINFER_EXL3_TARGET_O_K7_SMALL_M") == "0" &&
                        env("NINFER_EXL3_TARGET_GATEUP_SMALL_M") == "1" &&
                        env("NINFER_EXL3_TARGET_DOWN_SMALL_M") == "1",
                    "targetok7qual requires O flag0 and accepted K6 flags1");
            const auto ids = load_ids(prompt_file);
            require(ids.size() >= 512,
                    "targetok7qual prompt has fewer than 512 tokens");
            require(env_int("NINFER_E5A4_BLOCK", 8) == 8,
                    "targetok7qual requires NINFER_E5A4_BLOCK=8");
            std::vector<std::int64_t> prompt(ids.begin(), ids.begin() + 512);
            require(prompt.back() != kMaskToken,
                    "targetok7qual ctx512 ends in the draft mask token");
            std::ofstream out(out_path);
            require(out.good(), "cannot open targetok7qual report");
            run_target_o_k7_qualification(
                *target, *draft, prompt, oscar_requested, out);
            return 0;
        }
        // accept mode
        const std::string family = env("NINFER_E5A4_FAMILY");
        const std::string prompt_file = env("NINFER_E5A4_PROMPT_FILE");
        const std::string contexts = env("NINFER_E5A4_CONTEXTS");
        const int block_len = env_int("NINFER_E5A4_BLOCK", 8);
        const int reads = env_int("NINFER_E5A4_READS", 24);
        const int warmup = env_int("NINFER_E5A4_WARMUP", 2);
        const std::string summary_path = env("NINFER_E5A4_OUT");
        const std::string perread_path = env("NINFER_E5A4_PERREAD_OUT");
        const std::string hist_path = env("NINFER_E5A4_HIST_OUT");
        const std::string confidence_path =
            env("NINFER_E5A4_CONFIDENCE_OUT");
        const std::string confidence_policy_out_path =
            env("NINFER_E5A4_CONFIDENCE_POLICY_OUT");
        const bool target_projection_timing =
            env("NINFER_EXL3_TARGET_PROJECTION_TIMING") == "1";
        const std::string target_projection_timing_path =
            env("NINFER_EXL3_TARGET_PROJECTION_TIMING_OUT");
        require(!family.empty() && !prompt_file.empty() && !contexts.empty(), "E5A4 accept needs FAMILY/PROMPT_FILE/CONTEXTS");
        require(!summary_path.empty() && !perread_path.empty() && !hist_path.empty(), "E5A4 accept needs OUT/PERREAD_OUT/HIST_OUT");
        const auto all_ids = load_ids(prompt_file);
        std::ofstream summary(summary_path);
        std::ofstream perread(perread_path);
        std::ofstream hist(hist_path);
        std::ofstream confidence_out;
        if (!confidence_path.empty()) {
            require(env("NINFER_DFLASH2_POSITION_CONFIDENCE") == "1",
                    "confidence output requires POSITION_CONFIDENCE=1");
            confidence_out.open(confidence_path);
            require(confidence_out.good(),
                    "cannot open position confidence output");
            confidence_out
                << "family,ctx,read,position,selected_edge_score,"
                   "runner_up_edge_score,edge_margin,edge_normalized_margin,"
                   "selected_unary_score,best_unary_score,"
                   "runner_up_unary_score,unary_margin,"
                   "unary_normalized_margin,selected_candidate_rank,"
                   "label_known,label_matched,rejection_position,"
                   "attempted_block,accepted\n";
        }
        std::ofstream confidence_policy_out;
        if (!confidence_policy_out_path.empty()) {
            require(env("NINFER_E5A4_CONFIDENCE_POLICY") == "1",
                    "confidence policy output requires active policy");
            confidence_policy_out.open(confidence_policy_out_path);
            require(confidence_policy_out.good(),
                    "cannot open confidence policy output");
            confidence_policy_out
                << "family,ctx,read,burn_in,fallback,selected_block,decision_us,"
                   "expected_committed_b4,expected_committed_b6,expected_committed_b8,"
                   "efficiency_b4,efficiency_b6,efficiency_b8,accepted,committed,"
                   "rejection_position,attempted_verification_rows,replay_rows,"
                   "retained_rows,state_reconstruction_rows,draft_us,verify_us,round_us\n";
        }
        std::ofstream target_projection_out;
        if (target_projection_timing) {
            const std::string timing_verifier = env("NINFER_E5A4_VERIFIER");
            require(timing_verifier == "transaction" ||
                        timing_verifier == "transaction_retain",
                    "target projection timing requires a transaction verifier");
            require(!target_projection_timing_path.empty(),
                    "target projection timing requires TARGET_PROJECTION_TIMING_OUT");
            target_projection_out.open(target_projection_timing_path);
            require(target_projection_out.good(),
                    "cannot open target projection timing output");
            target_projection_out
                << "family,ctx,block,round,phase,layer,operator,rows,K,in_features,"
                   "out_features,topology,calls,microseconds\n";
        }
        require(summary.good() && perread.good() && hist.good(), "E5A4 cannot open outputs");
        summary << "family,ctx,block,reads,proposed,accepted,acc_rate,comm_per_read,median_acc,full_rate,zero_rate,draft_med_us,verify_med_us,read_wall_ms,control_ms,spec_tps,tgt_tps,speedup_seq,prefill_ms,correctness,oscar_full,ordinary_full,gdn,free_end,total_mem,protocol,total_emitted,total_target_decodes,initial_seed_us,verifier,transaction_setup_us,transaction_bytes,attempted_verification_rows,replay_rows,retained_rows,executed_target_rows,inference_total_ms,state_reconstruction_rows,control_prefill_ms,control_seed_us,control_ttft_ms,resident_ttft_ms,resident_request_ms,control_resident_ttft_ms,control_resident_request_ms,draft_prefill_submitted_rows,draft_prefill_encoded_rows,draft_prefill_skipped_rows,draft_prefill_overlap,draft_prefill_overlap_staging_bytes,draft_prefill_overlap_join_us,draft_prefill_overlap_guards,output_budget,useful_committed,surplus_retained,useful_spec_tps,spec_context_teardown_us\n";
        perread << "family,ctx,block,read,accepted,committed,rejpos,draft_us,verify_us,target_decodes,proposal_ids,round_us,verifier,attempted_verification_rows,replay_rows,retained_rows,executed_target_rows,state_reconstruction_rows,chosen_block\n";
        hist << "family,ctx,block,pos,count\n";
        std::stringstream ss(contexts);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            const int ctx = std::stoi(tok);
            require(ctx <= static_cast<int>(all_ids.size()), "E5A4 context longer than prompt file");
            std::vector<std::int64_t> prefix(all_ids.begin(), all_ids.begin() + ctx);
            if (prefix.back() == kMaskToken) prefix.pop_back();
            ConfigResult res = run_config(*target, *draft, prefix, block_len, reads,
                                          warmup, oscar_requested,
                                          handoff_profile.get());
            write_accept(family, static_cast<int>(prefix.size()), block_len, res, warmup,
                         summary, perread, hist);
            if (confidence_out.is_open())
                write_position_confidence(
                    family, static_cast<int>(prefix.size()), res,
                    confidence_out);
            if (confidence_policy_out.is_open())
                write_confidence_policy(
                    family, static_cast<int>(prefix.size()), res,
                    confidence_policy_out);
            if (target_projection_timing) {
                write_target_projection_timing(
                    family, static_cast<int>(prefix.size()), block_len, res,
                    target_projection_out);
            }
            summary.flush(); perread.flush(); hist.flush();
            if (confidence_out.is_open()) confidence_out.flush();
            if (confidence_policy_out.is_open()) confidence_policy_out.flush();
            // Optional correctness witness is written only after the measured request and normal result output.
            const auto exact_tokens_path = env("NINFER_E5A4_EXACT_TOKENS_OUT");
            if (!exact_tokens_path.empty()) {
                require(contexts.find(',') == std::string::npos && !std::filesystem::exists(exact_tokens_path),
                        "exact token witness requires one context and a new output");
                require(res.correctness && res.spec_tokens == res.control_tokens, "exact token witness mismatch");
                std::ofstream exact_tokens(exact_tokens_path);
                exact_tokens << "ordinal,spec_token,control_token\n";
                for (std::size_t i=0; i<res.spec_tokens.size(); ++i)
                    exact_tokens << i << ',' << res.spec_tokens[i] << ',' << res.control_tokens[i] << '\n';
                exact_tokens.flush(); require(exact_tokens.good(), "exact token witness write failed");
            }

        }
        if (ring_capture) { ring_capture->finish(); ring_capture.reset(); }
        if (handoff_profile) {
            handoff_profile->sample_memory(
                "accept_complete", -1, target.get(), draft.get(), nullptr);
            draft.reset();
            handoff_profile->sample_memory(
                "draft_destroyed", -1, target.get(), nullptr, nullptr);
            target.reset();
            handoff_profile->sample_memory(
                "target_destroyed", -1, nullptr, nullptr, nullptr);
            std::cout << "E5A4_HANDOFF_PROFILE_DONE prefix="
                      << handoff_profile_prefix << "\n";
        }
        std::cout << "E5A4_ACCEPT_DONE\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "E5A4 fatal: " << e.what() << "\n";
        return 1;
    }
}
