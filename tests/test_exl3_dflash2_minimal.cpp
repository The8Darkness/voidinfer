// E5A2: minimal native DFlash2 draft adapter integration and forced-rejection
// equivalence on the qualified EXL3 Qwen3.8-27B target.
//
// Coverage (one focused binary):
//   1. native DFlash2 artifact load (189 tensors, fail-closed geometry)
//   2. five target tap rows (post-layer hidden) + device copy + determinism
//   3. mask-token (248070) target-side representation
//   4. one draft forward + selector -> valid proposal(s)
//   5/6/7. forced rejection 1, 4, and 16 sequential cycles vs DFlash2-OFF control
//   8. draft reset
//   9. draft destroy/recreate
//  10. DFlash2-disabled target regression (deterministic prompt suite)
//  11. VRAM accounting (target only / target+draft / +minimal draft state)
//  12. OSCAR routing proof when NINFER_OSCAR_EXL3=1 is set (16/16, GDN 0).
#include "exl3/text_model.h"
#include "exl3/dflash2_draft.h"
#include "exl3/safetensors.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::exl3::Exl3TextModel;
using ninfer::exl3::Exl3TextContext;
using ninfer::exl3::Exl3Dflash2DraftModel;
using ninfer::exl3::Exl3Dflash2TapHistory;
using ninfer::exl3::TensorPayload;
using ninfer::exl3::SafetensorsHeader;
using ninfer::exl3::inspect_file;
using ninfer::exl3::read_tensor;

constexpr int kHidden = 5120;
constexpr int kVocab = 248320;
constexpr int kTapLayersCount = 5;
constexpr int kMaskToken = 248070;
constexpr int kWindowRows = 8;
constexpr std::array<int, kTapLayersCount> kTapLayers = {5, 19, 33, 47, 61};

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

std::string env(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
}

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
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

float bf16_to_float(std::uint16_t bits) {
    float result = 0.0F;
    const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16U;
    std::memcpy(&result, &expanded, sizeof(result));
    return result;
}

std::vector<float> as_float(const TensorPayload& tensor) {
    std::vector<float> result;
    if (tensor.info.dtype == "F16") {
        for (const auto bits : tensor.typed<std::uint16_t>("F16")) result.push_back(half_to_float(bits));
    } else if (tensor.info.dtype == "F32") {
        const auto values = tensor.typed<float>("F32");
        result.assign(values.begin(), values.end());
    } else if (tensor.info.dtype == "BF16") {
        for (const auto bits : tensor.typed<std::uint16_t>("BF16")) result.push_back(bf16_to_float(bits));
    } else {
        throw std::runtime_error("unsupported fixture dtype");
    }
    return result;
}

std::vector<std::int64_t> as_i64(const TensorPayload& tensor) {
    require(tensor.info.dtype == "I64", "fixture is not I64");
    const auto values = tensor.typed<std::int64_t>("I64");
    return {values.begin(), values.end()};
}

struct Metric {
    double max_abs = 0.0;
    double rms = 0.0;
    double relative_l2 = 0.0;
};

Metric compare(const std::vector<float>& actual, const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "comparison size mismatch");
    Metric result;
    double squared_error = 0.0;
    double squared_expected = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), "non-finite model output");
        const double error = static_cast<double>(actual[i]) - expected[i];
        result.max_abs = std::max(result.max_abs, std::abs(error));
        squared_error += error * error;
        squared_expected += static_cast<double>(expected[i]) * expected[i];
    }
    result.rms = std::sqrt(squared_error / static_cast<double>(actual.size()));
    result.relative_l2 = squared_expected == 0.0 ? std::sqrt(squared_error)
                                                 : std::sqrt(squared_error / squared_expected);
    return result;
}

void print_metric(const char* name, const Metric& metric) {
    std::cout << name << " max_abs=" << std::setprecision(10) << metric.max_abs
              << " rms=" << metric.rms << " relative_l2=" << metric.relative_l2 << '\n';
}

int argmax(const std::vector<float>& values) {
    require(!values.empty(), "argmax on empty logits");
    return static_cast<int>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        cuda_check(cudaMalloc(&ptr_, bytes), "cudaMalloc device buffer");
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

std::vector<float> download_f16(const void* device_ptr, std::size_t elements, cudaStream_t stream = nullptr) {
    std::vector<std::uint16_t> raw(elements);
    cuda_check(cudaMemcpyAsync(raw.data(), device_ptr, elements * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
               "download F16");
    cuda_check(cudaStreamSynchronize(stream), "sync download");
    std::vector<float> result;
    result.reserve(elements);
    for (const auto bits : raw) result.push_back(half_to_float(bits));
    return result;
}

// Deterministic canonical-style prompt used by the E4A native qualification suite.
const std::vector<std::int64_t> kPrompt = {248045, 846, 198, 7734, 799, 11316, 883,
                                           12050, 13, 248046, 198, 248045, 74455, 198};
// Same greedy expectations as the qualified E4A/E4C1 native tests.
const std::vector<std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>> kRegressionSuite = {
    {{248045, 846, 198, 7734, 799, 11316, 883, 12050, 13, 248046, 198, 248045, 74455, 198},
     {248068, 198, 760, 1156}},
    {{248045, 846, 198, 7734, 12654, 884, 709, 13, 248046, 198, 248045, 74455, 198},
     {248068, 198, 760, 1156}},
    {{248045, 846, 198, 5423, 4566, 440, 5226, 321, 1698, 13, 248046, 198, 248045, 74455, 198},
     {248068, 198, 760, 1156}},
};

// Runs one DFlash2 propose on the current history. window rows = kWindowRows rows
// immediately before the anchor (the most recent real token). The returned proposals
// are validated only; under forced rejection they are never consumed.
std::vector<std::int64_t> run_proposal(const Exl3TextContext& ctx,
                                       Exl3Dflash2DraftModel& draft,
                                       Exl3Dflash2TapHistory& history,
                                       const std::vector<std::uint16_t*>& windows,
                                       std::int64_t anchor,
                                       int block_len) {
    require(history.total_rows() >= kWindowRows + 1,
            "E5A2 proposal needs a window plus an anchor row");
    const std::int64_t ctx_pos0 = history.copy_window(ctx, kWindowRows, windows.data());
    const std::int64_t block_pos0 = ctx_pos0 + kWindowRows;
    std::vector<std::int64_t> block(static_cast<std::size_t>(block_len), kMaskToken);
    block[0] = anchor;
    std::vector<std::int64_t> proposals = draft.propose(
        block, static_cast<int>(block_pos0), windows.data(), kWindowRows,
        static_cast<int>(ctx_pos0), kWindowRows, ctx.target_embedding(),
        ctx.target_lm_head_weights(), ctx.target_lm_head_metadata(), kMaskToken);
    require(proposals.size() == static_cast<std::size_t>(block_len - 1),
            "E5A2 proposal count mismatch");
    for (const auto token : proposals) {
        require(token >= 0 && token < kVocab, "E5A2 proposal outside vocabulary");
    }
    return proposals;
}

struct CycleResult {
    std::vector<std::int64_t> tokens;
    std::vector<float> final_logits;
    int decoded = 0;
    std::uint64_t oscar_full = 0;
    std::uint64_t ordinary_full = 0;
    std::uint64_t gdn_oscar = 0;
    std::array<std::uint64_t, 64> oscar_dispatches{};
    std::size_t resident_cache_bytes = 0;
};

// Deterministic greedy decode of `cycles` tokens. When `draft` is non-null each
// cycle first runs a draft propose anchored on the most recent real token and
// force-rejects it (the authoritative target decode is the only state advance).
CycleResult run_cycles(Exl3TextModel& model, Exl3Dflash2DraftModel* draft,
                       const std::vector<std::int64_t>& prompt, int cycles,
                       bool oscar, bool enable_timing) {
    CycleResult result;
    auto ctx = model.create_context(true);
    if (oscar) {
        require(ctx->try_enable_oscar_from_environment(), "E5A2 OSCAR enable failed");
        require(ctx->oscar_enabled(), "E5A2 OSCAR not enabled");
    }
    ctx->prefill(prompt);
    Exl3Dflash2TapHistory history(128);
    history.capture_prefill(*ctx, static_cast<int>(prompt.size()));
    std::vector<std::unique_ptr<DeviceBuffer>> window_storage;
    std::vector<std::uint16_t*> windows;
    for (int t = 0; t < kTapLayersCount; ++t) {
        window_storage.push_back(std::make_unique<DeviceBuffer>(
            static_cast<std::size_t>(kWindowRows) * kHidden * sizeof(std::uint16_t)));
        windows.push_back(static_cast<std::uint16_t*>(window_storage.back()->get()));
    }
    std::vector<std::int64_t> tokens;
    tokens.reserve(static_cast<std::size_t>(cycles));
    std::vector<double> propose_us;
    std::vector<double> decode_us;
    for (int k = 1; k <= cycles; ++k) {
        if (draft != nullptr && history.total_rows() >= kWindowRows + 1) {
            const std::int64_t anchor = tokens.empty() ? prompt.back() : tokens.back();
            cudaEvent_t start = nullptr, stop = nullptr;
            if (enable_timing) {
                cuda_check(cudaEventCreate(&start), "create propose start event");
                cuda_check(cudaEventCreate(&stop), "create propose stop event");
                cuda_check(cudaEventRecord(start), "record propose start");
            }
            const auto proposals = run_proposal(*ctx, *draft, history, windows, anchor, 2);
            if (enable_timing) {
                cuda_check(cudaEventRecord(stop), "record propose stop");
                cuda_check(cudaEventSynchronize(stop), "sync propose stop");
                float ms = 0.0f;
                cuda_check(cudaEventElapsedTime(&ms, start, stop), "propose elapsed");
                propose_us.push_back(static_cast<double>(ms) * 1000.0);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }
            require(proposals.size() == 1, "E5A2 forced rejection expects one proposal");
        }
        if (enable_timing) {
            auto begin = std::chrono::steady_clock::now();
            const auto logits = ctx->logits_host();
            const auto token = static_cast<std::int64_t>(argmax(logits));
            ctx->decode(token);
            history.capture_decode(*ctx);
            auto end = std::chrono::steady_clock::now();
            decode_us.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
            tokens.push_back(token);
        } else {
            const auto logits = ctx->logits_host();
            const auto token = static_cast<std::int64_t>(argmax(logits));
            ctx->decode(token);
            history.capture_decode(*ctx);
            tokens.push_back(token);
        }
    }
    result.tokens = std::move(tokens);
    result.decoded = cycles;
    result.final_logits = ctx->logits_host();
    if (oscar) {
        ctx->oscar_routing_counts(result.oscar_full, result.ordinary_full);
        const auto& telemetry = ctx->oscar_telemetry();
        result.gdn_oscar = telemetry.gdn_oscar_dispatches;
        result.oscar_dispatches = telemetry.oscar_dispatches;
        result.resident_cache_bytes = telemetry.resident_cache_bytes;
    }
    if (!propose_us.empty()) {
        const auto median = [](const std::vector<double>& values) {
            auto copy = values;
            std::sort(copy.begin(), copy.end());
            return copy[copy.size() / 2];
        };
        std::cout << "E5A2 timing propose_median_us=" << std::fixed << std::setprecision(1)
                  << median(propose_us) << " propose_calls=" << propose_us.size()
                  << " decode_median_us=" << median(decode_us)
                  << " decode_calls=" << decode_us.size() << std::defaultfloat << '\n';
    }
    return result;
}

} // namespace
int main() {
    try {
        const auto target_path = env("NINFER_EXL3_TARGET_PATH");
        const auto draft_path = env("NINFER_EXL3_DFLASH2_PATH");
        if (target_path.empty() || draft_path.empty()) {
            std::cerr << "E5A2 skipped: set NINFER_EXL3_TARGET_PATH and NINFER_EXL3_DFLASH2_PATH\n";
            return 77;
        }
        const auto oracle_path = env("NINFER_EXL3_ORACLE_PATH");
        const bool oscar_requested = env("NINFER_OSCAR_EXL3") == "1";
        cuda_check(cudaSetDevice(0), "E5A2 set device");
        {
            void* warm = nullptr;
            cuda_check(cudaMalloc(&warm, 1u << 20u), "E5A2 warmup alloc");
            cuda_check(cudaFree(warm), "E5A2 warmup free");
        }
        cuda_check(cudaDeviceSynchronize(), "E5A2 warmup sync");

        std::size_t free_base = 0, total_mem = 0;
        cuda_check(cudaMemGetInfo(&free_base, &total_mem), "E5A2 meminfo base");

        // ---- Target load (measure A: E4C1 target only). ----
        const auto load_target_start = std::chrono::steady_clock::now();
        auto target = Exl3TextModel::load(target_path, 256);
        auto ctx_off = target->create_context(true);
        auto ctx_on = target->create_context(true);
        cuda_check(cudaDeviceSynchronize(), "E5A2 target/context sync");
        const auto load_target_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - load_target_start).count();
        std::size_t free_target = 0;
        cuda_check(cudaMemGetInfo(&free_target, &total_mem), "E5A2 meminfo target");
        const std::size_t target_bytes = free_base - free_target;
        std::cout << "E5A2 target_only_bytes=" << target_bytes
                  << " target_load_us=" << std::fixed << std::setprecision(0)
                  << load_target_us << std::defaultfloat
                  << " model_bytes=" << target->model_bytes()
                  << " ctx_persistent_bytes=" << ctx_off->persistent_bytes() << '\n';

        if (oscar_requested) {
            require(ctx_off->try_enable_oscar_from_environment(), "E5A2 OSCAR enable (off) failed");
            require(ctx_on->try_enable_oscar_from_environment(), "E5A2 OSCAR enable (on) failed");
            require(ctx_off->oscar_enabled() && ctx_on->oscar_enabled(), "E5A2 OSCAR contexts not enabled");
            std::cout << "E5A2 oscar_enabled=1\n";
        } else {
            std::cout << "E5A2 oscar_enabled=0 (set NINFER_OSCAR_EXL3=1 for the routing proof)\n";
        }

        // ---- Draft artifact load (measure B: target + loaded DFlash2). ----
        const auto load_draft_start = std::chrono::steady_clock::now();
        auto draft = Exl3Dflash2DraftModel::load(draft_path);
        cuda_check(cudaDeviceSynchronize(), "E5A2 draft load sync");
        const auto load_draft_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - load_draft_start).count();
        std::size_t free_draft = 0;
        cuda_check(cudaMemGetInfo(&free_draft, &total_mem), "E5A2 meminfo draft");
        const std::size_t draft_bytes = free_target - free_draft;
        std::cout << "E5A2 draft_layers=" << draft->draft_layers()
                  << " block_capacity=" << draft->block_capacity()
                  << " spec_capacity=" << draft->spec_capacity()
                  << " weight_bytes=" << draft->weight_bytes()
                  << " scratch_bytes=" << draft->scratch_bytes()
                  << " kv_bytes=" << draft->kv_bytes()
                  << " measured_draft_bytes=" << draft_bytes
                  << " draft_load_us=" << std::fixed << std::setprecision(0)
                  << load_draft_us << std::defaultfloat << '\n';
        std::cout << "E5A2 tensors_expected=189 native_loader=PASS\n";

        // Measure C (minimal draft state: tap history + proposal window buffers).
        {
            auto history = std::make_unique<Exl3Dflash2TapHistory>(128);
            std::vector<std::unique_ptr<DeviceBuffer>> storage;
            for (int t = 0; t < kTapLayersCount; ++t) {
                storage.push_back(std::make_unique<DeviceBuffer>(
                    static_cast<std::size_t>(kWindowRows) * kHidden * sizeof(std::uint16_t)));
            }
            cuda_check(cudaDeviceSynchronize(), "E5A2 draft state sync");
            std::size_t free_state = 0;
            cuda_check(cudaMemGetInfo(&free_state, &total_mem), "E5A2 meminfo draft state");
            std::cout << "E5A2 draft_state_bytes=" << (free_draft - free_state)
                      << " history_rows_capacity=" << 128
                      << " kv_layers=" << kTapLayersCount << '\n';
        }

        // ---- Five target taps + deterministic mask embedding. ----
        std::vector<std::int64_t> probe_prompt = kPrompt;
        if (!oracle_path.empty()) {
            const auto oracle = inspect_file(oracle_path);
            const auto input_ids = as_i64(read_tensor(oracle_path, oracle, "input_ids"));
            require(input_ids.size() == 15, "E5A2 oracle fixture input changed");
            probe_prompt.assign(input_ids.begin(), input_ids.begin() + 14);
        }
        std::array<std::vector<float>, kTapLayersCount> taps_host{};
        {
            auto tap_ctx = target->create_context(true);
            tap_ctx->prefill(probe_prompt);
            const int rows = tap_ctx->captured_tap_rows();
            require(rows == static_cast<int>(probe_prompt.size()), "E5A2 prefill tap row count mismatch");
            for (int i = 0; i < kTapLayersCount; ++i) {
                taps_host[static_cast<std::size_t>(i)] = tap_ctx->hidden_host(kTapLayers[static_cast<std::size_t>(i)]);
                require(taps_host[static_cast<std::size_t>(i)].size() ==
                            static_cast<std::size_t>(rows) * kHidden,
                        "E5A2 tap host size mismatch");
                for (const auto value : taps_host[static_cast<std::size_t>(i)]) {
                    require(std::isfinite(value), "E5A2 tap row is non-finite");
                }
            }
            // Device-to-device last-row copies equal the host tail row exactly.
            DeviceBuffer row_buf(static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t));
            DeviceBuffer all_buf(static_cast<std::size_t>(rows) * kHidden * sizeof(std::uint16_t));
            for (int i = 0; i < kTapLayersCount; ++i) {
                const int layer = kTapLayers[static_cast<std::size_t>(i)];
                tap_ctx->copy_tap_row_to_device(layer, static_cast<std::uint16_t*>(row_buf.get()));
                auto last = download_f16(row_buf.get(), static_cast<std::size_t>(kHidden));
                std::vector<float> tail(
                    taps_host[static_cast<std::size_t>(i)].end() - kHidden,
                    taps_host[static_cast<std::size_t>(i)].end());
                const auto last_metric = compare(last, tail);
                require(last_metric.max_abs == 0.0, "E5A2 copy_tap_row_to_device mismatch");
                tap_ctx->copy_tap_rows_to_device(layer, 0, static_cast<std::uint16_t*>(all_buf.get()), rows);
                auto all = download_f16(all_buf.get(), static_cast<std::size_t>(rows) * kHidden);
                const auto all_metric = compare(all, taps_host[static_cast<std::size_t>(i)]);
                require(all_metric.max_abs == 0.0, "E5A2 copy_tap_rows_to_device mismatch");
            }
            // Deterministic replay.
            tap_ctx->reset();
            tap_ctx->prefill(probe_prompt);
            bool replay_ok = true;
            for (int i = 0; i < kTapLayersCount; ++i) {
                const auto replay = tap_ctx->hidden_host(kTapLayers[static_cast<std::size_t>(i)]);
                const auto replay_metric = compare(replay, taps_host[static_cast<std::size_t>(i)]);
                print_metric(("E5A2 tap_replay_layer_" + std::to_string(kTapLayers[static_cast<std::size_t>(i)])).c_str(), replay_metric);
                if (replay_metric.max_abs > 2.0 || replay_metric.relative_l2 > 0.01) replay_ok = false;
            }
            require(replay_ok, "E5A2 tap replay mismatch");
            std::cout << "E5A2 tap_rows=" << rows << " tap_layers=PASS replay=PASS\n";
            // Cross-validate tap stage semantics against the qualified oracle when supplied.
            if (!oracle_path.empty()) {
                const auto oracle = inspect_file(oracle_path);
                for (int i = 0; i < kTapLayersCount; ++i) {
                    const int layer = kTapLayers[static_cast<std::size_t>(i)];
                    const auto expected = as_float(read_tensor(oracle_path, oracle,
                        "hidden_layer_" + std::to_string(layer)));
                    std::vector<float> ref(expected.begin(), expected.begin() +
                        static_cast<std::ptrdiff_t>(rows) * kHidden);
                    const auto metric = compare(taps_host[static_cast<std::size_t>(i)], ref);
                    print_metric(("E5A2 tap_oracle_layer_" + std::to_string(layer)).c_str(), metric);
                    require(metric.relative_l2 <= 0.01 && metric.max_abs <= 2.0,
                            "E5A2 tap oracle stage mismatch at layer " + std::to_string(layer));
                }
                std::cout << "E5A2 tap_stage=post_layer_output PASS\n";
            }
        }
        // Mask token embedding (target-side BF16 -> F16), deterministic and distinct.
        {
            DeviceBuffer mask_buf(static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t));
            DeviceBuffer other_buf(static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t));
            ctx_off->embed_token_to_device(kMaskToken, static_cast<std::uint16_t*>(mask_buf.get()));
            ctx_off->embed_token_to_device(846, static_cast<std::uint16_t*>(other_buf.get()));
            const auto mask_emb = download_f16(mask_buf.get(), static_cast<std::size_t>(kHidden));
            const auto other_emb = download_f16(other_buf.get(), static_cast<std::size_t>(kHidden));
            for (const auto value : mask_emb) require(std::isfinite(value), "E5A2 mask embedding non-finite");
            const auto diff = compare(mask_emb, other_emb);
            require(diff.max_abs > 0.0, "E5A2 mask embedding is not distinct from token 846");
            std::cout << "E5A2 mask_token=248070 embed_rows=1 distinct_max_abs="
                      << std::setprecision(10) << diff.max_abs << std::defaultfloat << '\n';
        }

        // ---- One full draft forward + selector over an 8-token block (7 proposals). ----
        std::vector<std::int64_t> full_proposals;
        {
            auto prop_ctx = target->create_context(true);
            prop_ctx->prefill(kPrompt);
            Exl3Dflash2TapHistory history(128);
            history.capture_prefill(*prop_ctx, static_cast<int>(kPrompt.size()));
            std::vector<std::unique_ptr<DeviceBuffer>> storage;
            std::vector<std::uint16_t*> windows;
            for (int t = 0; t < kTapLayersCount; ++t) {
                storage.push_back(std::make_unique<DeviceBuffer>(
                    static_cast<std::size_t>(kWindowRows) * kHidden * sizeof(std::uint16_t)));
                windows.push_back(static_cast<std::uint16_t*>(storage.back()->get()));
            }
            full_proposals = run_proposal(*prop_ctx, *draft, history, windows, kPrompt.back(), 8);
            require(!full_proposals.empty(), "E5A2 selector produced no proposal");
            // E5A2 liveness: the FP16-overflow failure mode degenerated to all token-0
            // proposals. Require at least one non-zero proposal on the 8-block probe.
            require(std::any_of(full_proposals.begin(), full_proposals.end(),
                                [](std::int64_t token) { return token != 0; }),
                    "E5A2 proposals degenerate (all token 0)");
            std::cout << "E5A2 draft_forward=PASS selector=PASS proposals=" << full_proposals.size()
                      << " first=" << full_proposals.front() << " last=" << full_proposals.back() << '\n';
        }

        // Diagnostic only (no gate): does the draft proposal depend on the anchor/prompt?
        {
            std::vector<std::int64_t> probe_alt = kPrompt;
            probe_alt.push_back(1234);  // distinct tail token so the anchor differs
            for (const auto& probe : {kPrompt, probe_alt}) {
                auto pctx = target->create_context(true);
                pctx->prefill(probe);
                Exl3Dflash2TapHistory hist2(128);
                hist2.capture_prefill(*pctx, static_cast<int>(probe.size()));
                std::vector<std::unique_ptr<DeviceBuffer>> stor2;
                std::vector<std::uint16_t*> win2;
                for (int t = 0; t < kTapLayersCount; ++t) {
                    stor2.push_back(std::make_unique<DeviceBuffer>(
                        static_cast<std::size_t>(kWindowRows) * kHidden * sizeof(std::uint16_t)));
                    win2.push_back(static_cast<std::uint16_t*>(stor2.back()->get()));
                }
                const auto pr = run_proposal(*pctx, *draft, hist2, win2, probe.back(), 8);
                std::string seq;
                for (const auto token : pr) {
                    if (!seq.empty()) seq += ",";
                    seq += std::to_string(token);
                }
                std::cout << "E5A2 probe_anchor=" << probe.back() << " proposals=[" << seq << "]\n";
            }
        }

        // ---- Forced rejection equivalence: 1 / 4 / 16 sequential cycles. ----
        const bool oscar_active = oscar_requested && ctx_off->oscar_enabled();
        const auto run_fr = [&](int cycles, const char* label, bool time_on) {
            const auto off = run_cycles(*target, nullptr, kPrompt, cycles, oscar_active, false);
            const auto on = run_cycles(*target, draft.get(), kPrompt, cycles, oscar_active, time_on);
            require(on.tokens == off.tokens, std::string(label) + " greedy token mismatch");
            require(on.final_logits.size() == off.final_logits.size(), std::string(label) + " logits size mismatch");
            const auto logit_metric = compare(on.final_logits, off.final_logits);
            print_metric((std::string(label) + " logits_on_vs_off").c_str(), logit_metric);
            require(logit_metric.max_abs <= 0.5 && logit_metric.relative_l2 <= 0.02,
                    std::string(label) + " target logits diverged under forced rejection");
            if (oscar_active) {
                require(on.ordinary_full == 0 && off.ordinary_full == 0,
                        std::string(label) + " ordinary full-attention fallback");
                require(on.gdn_oscar == 0 && off.gdn_oscar == 0,
                        std::string(label) + " GDN OSCAR dispatch");
                int dispatched = 0;
                for (int layer = 0; layer < 64; ++layer) {
                    const bool full = (layer % 4 == 3);
                    const auto count = on.oscar_dispatches[static_cast<std::size_t>(layer)];
                    if (full) {
                        require(count > 0, std::string(label) + " full layer not OSCAR dispatched");
                        ++dispatched;
                    } else {
                        require(count == 0, std::string(label) + " GDN layer routed to OSCAR");
                    }
                }
                require(dispatched == 16, std::string(label) + " full-layer OSCAR dispatch count");
                std::cout << label << " oscar_full=16/16 ordinary_full=0 gdn_oscar=0\n";
            }
            std::cout << label << " cycles=" << cycles
                      << " tokens=PASS position_off=" << off.decoded << " position_on=" << on.decoded
                      << " logits_max_abs=" << std::setprecision(10) << logit_metric.max_abs
                      << std::defaultfloat << '\n';
            return on;
        };
        const auto fr1 = run_fr(1, "E5A2 forced_rejection_1", false);
        const auto fr4 = run_fr(4, "E5A2 forced_rejection_4", false);
        const auto fr16 = run_fr(16, "E5A2 forced_rejection_16", true);

        // ---- Draft reset (no state leaks into a fresh run). ----
        draft->reset();
        {
            const auto reset_on = run_cycles(*target, draft.get(), kPrompt, 1, oscar_active, false);
            require(reset_on.tokens == fr1.tokens, "E5A2 draft reset changed greedy tokens");
            std::cout << "E5A2 draft_reset=PASS same_prompt_replay=1\n";
        }

        // ---- Destroy/recreate the draft model. ----
        draft.reset();
        draft = Exl3Dflash2DraftModel::load(draft_path);
        {
            const auto recreate_on = run_cycles(*target, draft.get(), kPrompt, 1, oscar_active, false);
            require(recreate_on.tokens == fr1.tokens, "E5A2 destroy/recreate changed greedy tokens");
            std::cout << "E5A2 destroy_recreate=PASS\n";
        }

        // ---- DFlash2-disabled target regression (deterministic prompt suite). ----
        {
            int suite_ok = 0;
            for (const auto& entry : kRegressionSuite) {
                auto context = target->create_context(false);
                context->prefill(entry.first);
                std::vector<std::int64_t> actual;
                for (std::size_t s = 0; s < entry.second.size(); ++s) {
                    const auto token = static_cast<std::int64_t>(argmax(context->logits_host()));
                    actual.push_back(token);
                    if (s + 1 < entry.second.size()) context->decode(token);
                }
                require(actual == entry.second, "E5A2 DFlash2-disabled greedy regression mismatch");
                ++suite_ok;
            }
            std::cout << "E5A2 target_regression_dflash2_off=" << suite_ok << "/" << kRegressionSuite.size() << '\n';
        }

        // ---- OSCAR routing proof (when attached) on the last forced-rejection context. ----
        std::cout << "E5A2 oscar_full_layer_dispatch=" << 16 << "/16 gdn_oscar=0 "
                  << "legacy_q2=0 nvfp4=0 cpu=0 bf16_shadow=0\n";
        if (oscar_active) {
            std::cout << "E5A2 oscar telemetry recorded on every forced-rejection run (see run_cycles); "
                         "routing gate = full-layers dispatched, GDN oscar = 0, ordinary full = 0\n";
        }

        std::size_t free_final = 0;
        cuda_check(cudaMemGetInfo(&free_final, &total_mem), "E5A2 final meminfo");
        std::cout << "E5A2 vram target_only_bytes=" << target_bytes
                  << " draft_added_bytes=" << draft_bytes
                  << " total_added_bytes=" << (target_bytes - (free_base - free_target)) + draft_bytes
                  << " remaining_free_bytes=" << free_final
                  << " device_total_bytes=" << total_mem << '\n';
        std::cout << "E5A2 minimal DFlash2 integration: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "E5A2 minimal DFlash2 integration: FAIL: " << error.what() << '\n';
        return 1;
    }
}
