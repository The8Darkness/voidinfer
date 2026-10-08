#include "exl3/linear_cuda.h"
#include "exl3/linear_reference.h"
#include "test_exl3_coherent_down_witness.h"
#include "test_exl3_coherent_wide_k6_witness.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;
using ninfer::exl3::Exl3CudaLinearMetadata;
using ninfer::exl3::Exl3CudaLinearWeights;
using ninfer::exl3::Exl3CudaLinearWorkspace;
using ninfer::exl3::Exl3LinearMetadata;
using ninfer::exl3::Exl3LinearTensors;
using ninfer::exl3::Exl3ReferenceOutput;
using ninfer::exl3::SafetensorsHeader;
using ninfer::exl3::TensorPayload;

constexpr const char* kTrellisName =
    "model.language_model.layers.5.mlp.gate_proj.trellis";
constexpr const char* kSuhName =
    "model.language_model.layers.5.mlp.gate_proj.suh";
constexpr const char* kSvhName =
    "model.language_model.layers.5.mlp.gate_proj.svh";
constexpr const char* kMul1Name =
    "model.language_model.layers.5.mlp.gate_proj.mul1";
constexpr const char* kDownPrefix =
    "model.language_model.layers.5.mlp.down_proj.";
constexpr const char* kDownK7Prefix =
    "model.language_model.layers.23.mlp.down_proj.";
constexpr const char* kCoherentOPrefix =
    "model.language_model.layers.23.self_attn.o_proj.";

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::string env_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
}

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

struct DeviceBuffer {
    void* pointer = nullptr;
    ~DeviceBuffer() { if (pointer != nullptr) { cudaFree(pointer); } }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer() = default;
};

template <typename T>
T* allocate_device(DeviceBuffer& owner, std::size_t count, const char* label) {
    cuda_check(cudaMalloc(&owner.pointer, count * sizeof(T)), label);
    return static_cast<T*>(owner.pointer);
}

TensorPayload load_indexed_tensor(const ninfer::exl3::IndexedSafetensors& collection,
                                  const char* name) {
    for (const auto& shard : collection.shards) {
        if (shard.header.find(name) != nullptr) {
            return ninfer::exl3::read_tensor(shard.path, shard.header, name);
        }
    }
    throw std::runtime_error(std::string("indexed tensor is absent: ") + name);
}

struct Metrics {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rms = 0.0;
    double relative_l2 = 0.0;
    double p99 = 0.0;
    double p999 = 0.0;
    std::size_t worst_index = 0;
    float actual_at_worst = 0.0f;
    float expected_at_worst = 0.0f;
};

Metrics compare(std::span<const std::uint16_t> actual,
                std::span<const std::uint16_t> expected) {
    require(actual.size() == expected.size(), "output size mismatch");
    std::vector<double> errors;
    errors.reserve(actual.size());
    double abs_sum = 0.0;
    double square_sum = 0.0;
    double expected_square_sum = 0.0;
    Metrics metrics;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const auto half_to_float = [](std::uint16_t bits) {
            const std::uint32_t sign = (bits & 0x8000u) << 16u;
            const std::uint32_t exponent = (bits >> 10u) & 0x1fu;
            const std::uint32_t fraction = bits & 0x03ffu;
            std::uint32_t value = sign;
            if (exponent == 0) {
                if (fraction != 0) {
                    std::uint32_t normalized = fraction;
                    std::uint32_t shift = 0;
                    while ((normalized & 0x0400u) == 0) {
                        normalized <<= 1u;
                        ++shift;
                    }
                    value |= (127u - 14u - shift) << 23u;
                    value |= (normalized & 0x03ffu) << 13u;
                }
            } else if (exponent == 0x1fu) {
                value |= 0x7f800000u | (fraction << 13u);
            } else {
                value |= (exponent + 112u) << 23u | (fraction << 13u);
            }
            float result;
            std::memcpy(&result, &value, sizeof(result));
            return result;
        };
        const float a = half_to_float(actual[i]);
        const float e = half_to_float(expected[i]);
        const double error = std::abs(static_cast<double>(a) - static_cast<double>(e));
        errors.push_back(error);
        abs_sum += error;
        square_sum += error * error;
        expected_square_sum += static_cast<double>(e) * static_cast<double>(e);
        if (error > metrics.max_abs) {
            metrics.max_abs = error;
            metrics.worst_index = i;
            metrics.actual_at_worst = a;
            metrics.expected_at_worst = e;
        }
    }
    std::sort(errors.begin(), errors.end());
    const auto percentile = [&errors](double p) {
        const auto index = std::min(errors.size() - 1,
                                    static_cast<std::size_t>(p * (errors.size() - 1)));
        return errors[index];
    };
    metrics.mean_abs = abs_sum / static_cast<double>(actual.size());
    metrics.rms = std::sqrt(square_sum / static_cast<double>(actual.size()));
    metrics.relative_l2 = expected_square_sum == 0.0
        ? std::sqrt(square_sum)
        : std::sqrt(square_sum / expected_square_sum);
    metrics.p99 = percentile(0.99);
    metrics.p999 = percentile(0.999);
    return metrics;
}

void print_metrics(const char* label, const Metrics& m) {
    std::cout << label << " max_abs=" << std::setprecision(10) << m.max_abs
              << " mean_abs=" << m.mean_abs
              << " rms=" << m.rms
              << " relative_l2=" << m.relative_l2
              << " p99=" << m.p99
              << " p999=" << m.p999
              << " worst_index=" << m.worst_index
              << " actual=" << m.actual_at_worst
              << " expected=" << m.expected_at_worst << '\n';
}

TensorPayload make_input(const TensorPayload& canonical, int rows) {
    const auto source = canonical.typed<std::uint16_t>("F16");
    require(rows >= 1 && rows <= 16, "test row count outside fixture plan");
    auto storage = std::make_shared<std::vector<std::byte>>(
        static_cast<std::size_t>(rows) * 5120u * sizeof(std::uint16_t));
    auto* destination = reinterpret_cast<std::uint16_t*>(storage->data());
    for (int row = 0; row < rows; ++row) {
        const int source_row = std::min(row, 13);
        std::memcpy(destination + static_cast<std::size_t>(row) * 5120u,
                    source.data() + static_cast<std::size_t>(source_row) * 5120u,
                    5120u * sizeof(std::uint16_t));
    }
    TensorPayload input;
    input.info.name = "linear_input";
    input.info.dtype = "F16";
    input.info.shape = {1, static_cast<std::uint64_t>(rows), 5120};
    input.info.data_end = storage->size();
    input.storage = std::move(storage);
    return input;
}

std::vector<std::uint16_t> expected_rows(std::span<const std::uint16_t> canonical,
                                         int rows) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(rows) * 17408u);
    for (int row = 0; row < rows; ++row) {
        const int source_row = std::min(row, 13);
        std::memcpy(result.data() + static_cast<std::size_t>(row) * 17408u,
                    canonical.data() + static_cast<std::size_t>(source_row) * 17408u,
                    17408u * sizeof(std::uint16_t));
    }
    return result;
}

template <typename Action>
void expect_invalid(const char* label, Action&& action) {
    try {
        action();
    } catch (const std::invalid_argument&) {
        std::cout << "negative PASS: " << label << '\n';
        return;
    }
    throw std::runtime_error(std::string("negative test unexpectedly succeeded: ") + label);
}

void run_negative_tests(Exl3CudaLinearWorkspace& workspace,
                        const Exl3CudaLinearWeights& weights,
                        const Exl3CudaLinearMetadata& metadata,
                        const std::uint16_t* input,
                        std::uint16_t* output) {
    expect_invalid("unsupported K", [&] {
        auto bad = metadata;
        bad.K = 6;
        workspace.forward(weights, bad, input, output, 1);
    });
    expect_invalid("wrong dimensions", [&] {
        auto bad = metadata;
        bad.in_features = 4096;
        workspace.forward(weights, bad, input, output, 1);
    });
    expect_invalid("unsupported bias", [&] {
        auto bad = metadata;
        bad.has_bias = true;
        workspace.forward(weights, bad, input, output, 1);
    });
    expect_invalid("unsupported flags", [&] {
        auto bad = metadata;
        bad.mul1 = false;
        workspace.forward(weights, bad, input, output, 1);
    });
    expect_invalid("missing trellis device buffer", [&] {
        auto bad = weights;
        bad.trellis = nullptr;
        workspace.forward(bad, metadata, input, output, 1);
    });
    expect_invalid("missing mul1 device buffer", [&] {
        auto bad = weights;
        bad.mul1 = nullptr;
        workspace.forward(bad, metadata, input, output, 1);
    });
    expect_invalid("invalid launch shape", [&] {
        workspace.forward(weights, metadata, input, output, 17);
    });
    std::cout << "CUDA negative tests: PASS\n";
}

double benchmark(Exl3CudaLinearWorkspace& workspace,
                 const Exl3CudaLinearWeights& weights,
                 const Exl3CudaLinearMetadata& metadata,
                 const std::uint16_t* input,
                 std::uint16_t* output,
                 int rows) {
    const int warmup = env_or_empty("NINFER_EXL3_WARMUP").empty()
        ? 10 : std::stoi(env_or_empty("NINFER_EXL3_WARMUP"));
    const int iterations = env_or_empty("NINFER_EXL3_ITERATIONS").empty()
        ? 40 : std::stoi(env_or_empty("NINFER_EXL3_ITERATIONS"));
    require(warmup >= 0 && iterations > 0, "invalid EXL3 benchmark iteration settings");
    for (int i = 0; i < warmup; ++i) {
        workspace.forward(weights, metadata, input, output, rows);
    }
    cuda_check(cudaDeviceSynchronize(), "EXL3 warmup synchronize");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");
    cuda_check(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < iterations; ++i) {
        workspace.forward(weights, metadata, input, output, rows);
    }
    cuda_check(cudaEventRecord(stop), "cudaEventRecord stop");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float milliseconds = 0.0f;
    cuda_check(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(milliseconds) * 1000.0 / iterations;
}

void run_fast_small_m_down_operator(const std::filesystem::path& model_dir) {
    constexpr int input_features = 17408;
    constexpr int output_features = 5120;
    const bool coherent =
        env_or_empty("NINFER_EXL3_TEST_COHERENT_DOWN_K6") == "1";
    const auto collection = ninfer::exl3::inspect_indexed_directory(model_dir);
    std::vector<TensorPayload> host;
    for (const char* suffix : {"trellis", "suh", "svh", "mul1"}) {
        host.push_back(load_indexed_tensor(collection,
            (std::string(kDownPrefix) + suffix).c_str()));
    }
    require(host[1].bytes().size() == input_features * sizeof(std::uint16_t) &&
                host[2].bytes().size() == output_features * sizeof(std::uint16_t) &&
                host[3].bytes().size() == sizeof(std::int32_t),
            "down projection auxiliary weight extent mismatch");
    DeviceBuffer d_trellis, d_suh, d_svh, d_mul1, d_input, d_control, d_candidate;
    auto upload = [](DeviceBuffer& buffer, const TensorPayload& payload,
                     const char* label) {
        auto* pointer = allocate_device<std::byte>(buffer, payload.bytes().size(), label);
        cuda_check(cudaMemcpy(pointer, payload.bytes().data(), payload.bytes().size(),
                              cudaMemcpyHostToDevice), label);
        return pointer;
    };
    const Exl3CudaLinearWeights weights{
        reinterpret_cast<const std::uint16_t*>(upload(d_trellis, host[0], "down trellis")),
        reinterpret_cast<const std::uint16_t*>(upload(d_suh, host[1], "down suh")),
        reinterpret_cast<const std::uint16_t*>(upload(d_svh, host[2], "down svh")),
        reinterpret_cast<const std::int32_t*>(upload(d_mul1, host[3], "down mul1"))};
    auto* input = allocate_device<std::uint16_t>(d_input, 8u * input_features, "down input");
    auto* control_output = allocate_device<std::uint16_t>(
        d_control, 8u * output_features, "down control output");
    constexpr std::size_t guard_elements = 32;
    auto* candidate_output = allocate_device<std::uint16_t>(
        d_candidate, 8u * output_features + guard_elements, "down candidate output");
    std::vector<std::uint16_t> input_bits(8u * input_features);
    for (std::size_t index = 0; index < input_bits.size(); ++index) {
        const std::uint16_t magnitude = static_cast<std::uint16_t>(
            0x2800u + ((index * 73u + index / input_features * 31u) % 0x800u));
        input_bits[index] = static_cast<std::uint16_t>(
            magnitude | (((index / 19u) & 1u) ? 0x8000u : 0u));
    }
    const auto down_witness = coherent_down_witness::evaluate(
        input_bits, host[1].typed<std::uint16_t>("F16"),
        host[2].typed<std::uint16_t>("F16"),
        host[0].typed<std::uint16_t>("I16"),
        static_cast<std::uint32_t>(host[3].typed<std::int32_t>("I32")[0]), 0);
    std::size_t scalar_split_differences = 0;
    for (int j = 0; j < coherent_down_witness::width; ++j)
        scalar_split_differences +=
            down_witness.scalar_fp32_output[j] !=
            down_witness.split5_fp32_output[j];
    std::cout << "COHERENT_DOWN_WITNESS group=" << coherent_down_witness::group
              << " row=0 scalar_split_output_bits=" << scalar_split_differences
              << " (CPU arithmetic profiles; not CUDA MMA parity)\n";
    cuda_check(cudaMemcpy(input, input_bits.data(), input_bits.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice), "down input upload");
    const Exl3CudaLinearMetadata metadata{
        input_features, output_features, 6, false, true, false};
    _putenv_s("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16_M1", coherent ? "0" : "1");
    _putenv_s("NINFER_EXL3_FAST_FP16_M2_8", "0");
    _putenv_s("NINFER_EXL3_FAST_FP16_M2_8_DOWN_K6", "0");
    _putenv_s("NINFER_EXL3_FAST_FP16_M2_8_FUSED_DOWN_K6", "0");
    _putenv_s("NINFER_EXL3_COHERENT_DOWN_K6", "0");
    Exl3CudaLinearWorkspace control(input_features, output_features, 8,
                                    false, false, true);
    const bool fused_down =
        !coherent && env_or_empty("NINFER_EXL3_TEST_FAST_SMALL_M_FUSED_DOWN") == "1";
    if (coherent)
        _putenv_s("NINFER_EXL3_COHERENT_DOWN_K6", "1");
    else if (fused_down)
        _putenv_s("NINFER_EXL3_FAST_FP16_M2_8_FUSED_DOWN_K6", "1");
    else
        _putenv_s("NINFER_EXL3_FAST_FP16_M2_8", "1");
    Exl3CudaLinearWorkspace candidate(input_features, output_features, 8,
                                      false, false, true);
    using Admission = ninfer::exl3::Exl3CudaLinearAdmission;
    const auto calls = [&]() {
        return coherent
            ? Exl3CudaLinearWorkspace::process_coherent_down_k6_calls_for_test()
            : fused_down
            ? Exl3CudaLinearWorkspace::process_fast_fp16_m2_8_fused_down_calls_for_test()
            : Exl3CudaLinearWorkspace::process_fast_fp16_m2_8_down_calls_for_test();
    };
    const auto before = calls();
    const auto before_rows =
        Exl3CudaLinearWorkspace::process_coherent_down_k6_rows_for_test();
    std::vector<coherent_down_witness::Result> row_oracles;
    if (coherent) {
        require(candidate.coherent_down_k6_resident_capacity_for_test() > 0,
                "coherent K6 down has no resident block capacity");
        for (int row = 0; row < 8; ++row)
            row_oracles.push_back(coherent_down_witness::evaluate(
                input_bits, host[1].typed<std::uint16_t>("F16"),
                host[2].typed<std::uint16_t>("F16"),
                host[0].typed<std::uint16_t>("I16"),
                static_cast<std::uint32_t>(
                    host[3].typed<std::int32_t>("I32")[0]), row));
        require(std::string(candidate.dispatch_name(metadata, 1,
                    Admission::target_continuation_down)) ==
                    "coherent_down_k6_shared_rows_n8_split5",
                "coherent correction M1 admission did not select shared core");
    }
    const std::vector<int> row_counts = coherent
        ? std::vector<int>{1, 2, 3, 4, 7, 8}
        : std::vector<int>{2, 4, 8};
    std::array<std::vector<std::uint16_t>, 8> coherent_row_snapshots;
    for (int rows : row_counts) {
        const auto admission = rows == 1 ? Admission::ordinary
                                         : Admission::target_continuation_down;
        cuda_check(cudaMemset(candidate_output, 0xa5,
            (8u * output_features + guard_elements) * sizeof(std::uint16_t)),
            "initialize coherent down output and tail guard");
        control.forward(weights, metadata, input, control_output, rows,
                        nullptr, admission);
        candidate.forward(weights, metadata, input, candidate_output, rows,
                          nullptr, admission);
        cuda_check(cudaDeviceSynchronize(), "down differential synchronization");
        std::vector<std::uint16_t> expected(static_cast<std::size_t>(rows) * output_features);
        std::vector<std::uint16_t> actual(expected.size());
        cuda_check(cudaMemcpy(expected.data(), control_output,
                              expected.size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost), "down control download");
        cuda_check(cudaMemcpy(actual.data(), candidate_output,
                              actual.size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost), "down candidate download");
        const auto metrics = compare(actual, expected);
        print_metrics((std::string("FAST_SMALL_M_DOWN M=") + std::to_string(rows)).c_str(), metrics);
        std::size_t mismatches = 0;
        for (std::size_t index = 0; index < actual.size(); ++index) {
            mismatches += actual[index] != expected[index];
        }
        std::cout << "FAST_SMALL_M_DOWN M=" << rows << " bit_mismatches=" << mismatches
                  << " control_dispatch=" << control.dispatch_name(metadata, rows,
                       admission)
                  << " candidate_dispatch=" << candidate.dispatch_name(metadata, rows,
                       admission) << '\n';
        if (coherent) {
            for (int row = 0; row < rows; ++row) {
                const auto first = actual.begin() +
                    static_cast<std::size_t>(row) * output_features;
                auto& snapshot = coherent_row_snapshots[row];
                if (snapshot.empty())
                    snapshot.assign(first, first + output_features);
                else
                    require(std::equal(snapshot.begin(), snapshot.end(), first),
                            "coherent K6 down changed a row with verifier width");
            }
            require(std::string(candidate.dispatch_name(metadata, rows, admission)) ==
                        "coherent_down_k6_shared_rows_n8_split5" &&
                        std::string(control.dispatch_name(metadata, rows, admission)) !=
                        "coherent_down_k6_shared_rows_n8_split5",
                    "coherent K6 down dispatch admission/fallback mismatch");
            std::vector<std::uint16_t> guard(
                8u * output_features + guard_elements - expected.size());
            cuda_check(cudaMemcpy(guard.data(), candidate_output + expected.size(),
                                  guard.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "coherent K6 down tail guard download");
            require(std::all_of(guard.begin(), guard.end(),
                        [](std::uint16_t value) { return value == 0xa5a5u; }),
                    "coherent K6 down wrote an inactive tail row or guard");
            double error_sq = 0.0, norm_sq = 0.0, oracle_max_abs = 0.0;
            for (int row = 0; row < rows; ++row) {
                for (int j = 0; j < coherent_down_witness::width; ++j) {
                    const int column = coherent_down_witness::group *
                        coherent_down_witness::width + j;
                    const double value = coherent_down_witness::from_half(
                        actual[static_cast<std::size_t>(row) * output_features + column]);
                    const double reference = row_oracles[row].fp64_output[j];
                    const double error = value - reference;
                    require(std::isfinite(value) && std::isfinite(reference),
                            "coherent K6 down independent oracle nonfinite");
                    error_sq += error * error;
                    norm_sq += reference * reference;
                    oracle_max_abs = std::max(oracle_max_abs, std::abs(error));
                }
            }
            const double relative_l2 = std::sqrt(error_sq / std::max(norm_sq, 1e-30));
            std::cout << "COHERENT_DOWN_K6 M=" << rows
                      << " oracle_group=" << coherent_down_witness::group
                      << " oracle_max_abs=" << oracle_max_abs
                      << " oracle_relative_l2=" << relative_l2
                      << " resident_capacity="
                      << candidate.coherent_down_k6_resident_capacity_for_test()
                      << " shared_bytes=12288\n";
            require(oracle_max_abs <= 0.02 && relative_l2 <= 0.002,
                    "coherent K6 down exceeds independent FP64 operator bound");
        }
        double control_oracle_max = 0.0, candidate_oracle_max = 0.0;
        for (int j = 0; j < coherent_down_witness::width; ++j) {
            const auto index = coherent_down_witness::group *
                coherent_down_witness::width + j;
            const double mathematical = down_witness.fp64_output[j];
            control_oracle_max = std::max(control_oracle_max,
                std::abs(static_cast<double>(coherent_down_witness::from_half(
                    expected[index])) - mathematical));
            candidate_oracle_max = std::max(candidate_oracle_max,
                std::abs(static_cast<double>(coherent_down_witness::from_half(
                    actual[index])) - mathematical));
        }
        std::cout << "COHERENT_DOWN_ORACLE M=" << rows
                  << " row=0 group=" << coherent_down_witness::group
                  << " control_max_abs=" << control_oracle_max
                  << " candidate_max_abs=" << candidate_oracle_max << '\n';
        require(std::isfinite(control_oracle_max) &&
                    std::isfinite(candidate_oracle_max),
                "down independent packed-weight oracle found nonfinite output");
        require(std::isfinite(metrics.max_abs) && std::isfinite(metrics.relative_l2),
                "fast small-M down emitted nonfinite output");
    }
    if (coherent) {
        std::vector<std::uint16_t> scalar(output_features), correction(output_features);
        candidate.forward(weights, metadata, input, candidate_output, 1,
                          nullptr, Admission::ordinary);
        cuda_check(cudaMemcpy(scalar.data(), candidate_output,
                              scalar.size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "coherent K6 down scalar M1 download");
        candidate.forward(weights, metadata, input, candidate_output, 1,
                          nullptr, Admission::target_continuation_down);
        cuda_check(cudaMemcpy(correction.data(), candidate_output,
                              correction.size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "coherent K6 down correction M1 download");
        require(scalar == correction,
                "coherent K6 down scalar and correction M1 arithmetic differ");
    }
    const auto after = calls();
    require(after - before == row_counts.size() + (coherent ? 2u : 0u),
            "selected down path was not dispatched for each row count");
    if (coherent)
        require(Exl3CudaLinearWorkspace::process_coherent_down_k6_rows_for_test() -
                    before_rows == 27,
                "coherent K6 down dispatched row count mismatch");
    if (fused_down) {
        const auto control_us = benchmark(control, weights, metadata, input,
            control_output, 8);
        const auto candidate_us = benchmark(candidate, weights, metadata, input,
            candidate_output, 8);
        std::cout << "FAST_SMALL_M_FUSED_DOWN_TIMING M=8 control_us=" <<
            control_us << " candidate_us=" << candidate_us << '\n';
    }
    std::cout << "FAST_SMALL_M_DOWN_OPERATOR OBSERVED dispatch_calls=" << after - before << '\n';
}

void run_coherent_o_k7_operator(const std::filesystem::path& model_dir) {
    constexpr int in_features=6144,out_features=5120;
    constexpr std::size_t guard=64;
    using Admission=ninfer::exl3::Exl3CudaLinearAdmission;
    const auto collection=ninfer::exl3::inspect_indexed_directory(model_dir);
    std::vector<TensorPayload> host;
    for(const char* suffix:{"trellis","suh","svh","mul1"})
        host.push_back(load_indexed_tensor(collection,
            (std::string(kCoherentOPrefix)+suffix).c_str()));
    require(host[0].bytes().size()==static_cast<std::size_t>(in_features/16)*
                (out_features/16)*7u*16u*sizeof(std::uint16_t) &&
            host[1].bytes().size()==in_features*sizeof(std::uint16_t) &&
            host[2].bytes().size()==out_features*sizeof(std::uint16_t) &&
            host[3].bytes().size()==sizeof(std::int32_t),
        "coherent K7 O real weight extent mismatch");
    DeviceBuffer d_trellis,d_suh,d_svh,d_mul1,d_input,d_control,d_candidate;
    const auto upload=[](DeviceBuffer& owner,const TensorPayload& tensor,
                         const char* label) {
        auto* ptr=allocate_device<std::byte>(owner,tensor.bytes().size(),label);
        cuda_check(cudaMemcpy(ptr,tensor.bytes().data(),tensor.bytes().size(),
                              cudaMemcpyHostToDevice),label);
        return ptr;
    };
    const Exl3CudaLinearWeights weights{
        reinterpret_cast<const std::uint16_t*>(upload(d_trellis,host[0],"K7 O trellis")),
        reinterpret_cast<const std::uint16_t*>(upload(d_suh,host[1],"K7 O suh")),
        reinterpret_cast<const std::uint16_t*>(upload(d_svh,host[2],"K7 O svh")),
        reinterpret_cast<const std::int32_t*>(upload(d_mul1,host[3],"K7 O mul1"))};
    auto* input=allocate_device<std::uint16_t>(d_input,8u*in_features,"K7 O input");
    auto* control=allocate_device<std::uint16_t>(d_control,8u*out_features,"K7 O control");
    auto* candidate=allocate_device<std::uint16_t>(d_candidate,
        8u*out_features+guard,"K7 O candidate");
    std::vector<std::uint16_t> input_bits(8u*in_features);
    for(std::size_t i=0;i<input_bits.size();++i) {
        const auto magnitude=static_cast<std::uint16_t>(
            0x2800u+((i*73u+i/in_features*31u)%0x800u));
        input_bits[i]=static_cast<std::uint16_t>(
            magnitude|(((i/19u)&1u)?0x8000u:0u));
    }
    cuda_check(cudaMemcpy(input,input_bits.data(),input_bits.size()*sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),"K7 O input upload");
    const Exl3CudaLinearMetadata metadata{in_features,out_features,7,false,true,false};
    _putenv_s("NINFER_EXL3_COHERENT_O_K7","0");
    Exl3CudaLinearWorkspace old(in_features,out_features,8,
        false,false,false,true);
    _putenv_s("NINFER_EXL3_COHERENT_O_K7","1");
    Exl3CudaLinearWorkspace shared(in_features,out_features,8,
        false,false,false,true);
    require(shared.coherent_o_k7_resident_capacity_for_test()>0,
        "coherent K7 O has no resident block capacity");
    const auto before_calls=Exl3CudaLinearWorkspace::
        process_coherent_o_k7_calls_for_test();
    const auto before_rows=Exl3CudaLinearWorkspace::
        process_coherent_o_k7_rows_for_test();
    std::vector<coherent_down_witness::Result> oracle;
    for(int row=0;row<8;++row)
        oracle.push_back(coherent_down_witness::evaluate_o_k7(
            input_bits,host[1].typed<std::uint16_t>("F16"),
            host[2].typed<std::uint16_t>("F16"),
            host[0].typed<std::uint16_t>("I16"),
            static_cast<std::uint32_t>(host[3].typed<std::int32_t>("I32")[0]),row));
    std::array<std::vector<std::uint16_t>,8> row_snapshots;
    for(int rows:{1,2,3,4,7,8}) {
        const auto admission=rows==1?Admission::ordinary:
            Admission::target_continuation_o;
        require(std::string(shared.dispatch_name(metadata,rows,admission))==
                    "coherent_o_k7_shared_rows_n8_split5" &&
                std::string(old.dispatch_name(metadata,rows,admission))!=
                    "coherent_o_k7_shared_rows_n8_split5",
            "coherent K7 O dispatch or fallback mismatch");
        cuda_check(cudaMemset(candidate,0xa5,
            (8u*out_features+guard)*sizeof(std::uint16_t)),
            "K7 O tail canary");
        old.forward(weights,metadata,input,control,rows,nullptr,admission);
        shared.forward(weights,metadata,input,candidate,rows,nullptr,admission);
        cuda_check(cudaDeviceSynchronize(),"K7 O differential synchronize");
        std::vector<std::uint16_t> reference(rows*out_features),actual(rows*out_features);
        cuda_check(cudaMemcpy(reference.data(),control,reference.size()*sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),"K7 O control download");
        cuda_check(cudaMemcpy(actual.data(),candidate,actual.size()*sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),"K7 O shared download");
        std::vector<std::uint16_t> tail(8u*out_features+guard-actual.size());
        cuda_check(cudaMemcpy(tail.data(),candidate+actual.size(),
            tail.size()*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
            "K7 O tail download");
        require(std::all_of(tail.begin(),tail.end(),
            [](std::uint16_t bits){return bits==0xa5a5u;}),
            "coherent K7 O wrote inactive row or tail");
        double error_sq=0.0,norm_sq=0.0,max_abs=0.0;
        for(int row=0;row<rows;++row) {
            const auto first=actual.begin()+static_cast<std::size_t>(row)*out_features;
            auto& snapshot=row_snapshots[row];
            if(snapshot.empty())snapshot.assign(first,first+out_features);
            else require(std::equal(snapshot.begin(),snapshot.end(),first),
                "coherent K7 O row changed with verifier width");
            for(int j=0;j<coherent_down_witness::width;++j) {
                const int col=coherent_down_witness::group*
                    coherent_down_witness::width+j;
                const double value=coherent_down_witness::from_half(
                    actual[static_cast<std::size_t>(row)*out_features+col]);
                const double expected=oracle[row].fp64_output[j];
                require(std::isfinite(value)&&std::isfinite(expected),
                    "coherent K7 O independent FP64 nonfinite");
                const double error=value-expected;
                error_sq+=error*error;norm_sq+=expected*expected;
                max_abs=std::max(max_abs,std::abs(error));
            }
        }
        const auto rel_l2=std::sqrt(error_sq/std::max(norm_sq,1e-30));
        const auto metrics=compare(actual,reference);
        require(max_abs<=0.02 && rel_l2<=0.002,
            "coherent K7 O exceeds independent FP64 operator bound");
        std::cout<<"COHERENT_O_K7 M="<<rows
                 <<" oracle_max_abs="<<max_abs
                 <<" oracle_relative_l2="<<rel_l2
                 <<" control_max_abs="<<metrics.max_abs
                 <<" resident_capacity="<<shared.coherent_o_k7_resident_capacity_for_test()
                 <<" shared_bytes=12800\n";
    }
    std::vector<std::uint16_t> scalar(out_features),correction(out_features);
    shared.forward(weights,metadata,input,candidate,1,nullptr,Admission::ordinary);
    cuda_check(cudaMemcpy(scalar.data(),candidate,scalar.size()*sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),"K7 O scalar download");
    shared.forward(weights,metadata,input,candidate,1,nullptr,
                   Admission::target_continuation_o);
    cuda_check(cudaMemcpy(correction.data(),candidate,
        correction.size()*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
        "K7 O correction download");
    require(scalar==correction,"coherent K7 O scalar/correction M1 mismatch");
    require(Exl3CudaLinearWorkspace::process_coherent_o_k7_calls_for_test()-
                before_calls==8 &&
            Exl3CudaLinearWorkspace::process_coherent_o_k7_rows_for_test()-
                before_rows==27,
        "coherent K7 O dispatch counts or rows mismatch");
    std::cout<<"COHERENT_O_K7_OPERATOR PASS calls=8 rows=27\n";
}

void run_coherent_down_k7_operator(const std::filesystem::path& model_dir) {
    constexpr int in_features=17408,out_features=5120;
    constexpr std::size_t guard=64;
    using Admission=ninfer::exl3::Exl3CudaLinearAdmission;
    const auto collection=ninfer::exl3::inspect_indexed_directory(model_dir);
    std::vector<TensorPayload> host;
    for(const char* suffix:{"trellis","suh","svh","mul1"})
        host.push_back(load_indexed_tensor(collection,
            (std::string(kDownK7Prefix)+suffix).c_str()));
    require(host[0].bytes().size()==static_cast<std::size_t>(in_features/16)*
                (out_features/16)*7u*16u*sizeof(std::uint16_t) &&
            host[1].bytes().size()==in_features*sizeof(std::uint16_t) &&
            host[2].bytes().size()==out_features*sizeof(std::uint16_t) &&
            host[3].bytes().size()==sizeof(std::int32_t),
        "coherent K7 down real weight extent mismatch");
    DeviceBuffer d_trellis,d_suh,d_svh,d_mul1,d_input,d_control,d_candidate;
    const auto upload=[](DeviceBuffer& owner,const TensorPayload& tensor,
                         const char* label) {
        auto* ptr=allocate_device<std::byte>(owner,tensor.bytes().size(),label);
        cuda_check(cudaMemcpy(ptr,tensor.bytes().data(),tensor.bytes().size(),
                              cudaMemcpyHostToDevice),label);
        return ptr;
    };
    const Exl3CudaLinearWeights weights{
        reinterpret_cast<const std::uint16_t*>(upload(d_trellis,host[0],"K7 down trellis")),
        reinterpret_cast<const std::uint16_t*>(upload(d_suh,host[1],"K7 down suh")),
        reinterpret_cast<const std::uint16_t*>(upload(d_svh,host[2],"K7 down svh")),
        reinterpret_cast<const std::int32_t*>(upload(d_mul1,host[3],"K7 down mul1"))};
    auto* input=allocate_device<std::uint16_t>(d_input,8u*in_features,"K7 down input");
    auto* control=allocate_device<std::uint16_t>(d_control,8u*out_features,"K7 down control");
    auto* candidate=allocate_device<std::uint16_t>(d_candidate,
        8u*out_features+guard,"K7 down candidate");
    std::vector<std::uint16_t> input_bits(8u*in_features);
    for(std::size_t i=0;i<input_bits.size();++i) {
        const auto magnitude=static_cast<std::uint16_t>(
            0x2800u+((i*73u+i/in_features*31u)%0x800u));
        input_bits[i]=static_cast<std::uint16_t>(
            magnitude|(((i/19u)&1u)?0x8000u:0u));
    }
    cuda_check(cudaMemcpy(input,input_bits.data(),input_bits.size()*sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),"K7 down input upload");
    const Exl3CudaLinearMetadata metadata{in_features,out_features,7,false,true,false};
    _putenv_s("NINFER_EXL3_COHERENT_DOWN_K7","0");
    Exl3CudaLinearWorkspace old(in_features,out_features,8,
        false,false,true,false);
    _putenv_s("NINFER_EXL3_COHERENT_DOWN_K7","1");
    Exl3CudaLinearWorkspace shared(in_features,out_features,8,
        false,false,true,false);
    require(shared.coherent_down_k7_shared_bytes_for_test()==12800,
        "coherent K7 down shared resource declaration mismatch");
    require(shared.coherent_down_k7_resident_capacity_for_test()>0,
        "coherent K7 down has no resident block capacity");
    require(std::string(shared.dispatch_name(
                Exl3CudaLinearMetadata{in_features,out_features,6,false,true,false},
                1,Admission::ordinary))!=
                "coherent_down_k7_shared_rows_n8_split5" &&
            std::string(shared.dispatch_name(metadata,9,Admission::ordinary))!=
                "coherent_down_k7_shared_rows_n8_split5",
        "coherent K7 down escaped its metadata or row contract");
    const auto before_calls=Exl3CudaLinearWorkspace::
        process_coherent_down_k7_calls_for_test();
    const auto before_rows=Exl3CudaLinearWorkspace::
        process_coherent_down_k7_rows_for_test();
    std::vector<coherent_down_witness::Result> oracle;
    for(int row=0;row<8;++row)
        oracle.push_back(coherent_down_witness::evaluate_down_k7(
            input_bits,host[1].typed<std::uint16_t>("F16"),
            host[2].typed<std::uint16_t>("F16"),
            host[0].typed<std::uint16_t>("I16"),
            static_cast<std::uint32_t>(host[3].typed<std::int32_t>("I32")[0]),row));
    std::array<std::vector<std::uint16_t>,8> row_snapshots;
    for(int rows:{1,2,3,4,7,8}) {
        const auto admission=rows==1?Admission::ordinary:
            Admission::target_continuation_down;
        require(std::string(shared.dispatch_name(metadata,rows,admission))==
                    "coherent_down_k7_shared_rows_n8_split5" &&
                std::string(old.dispatch_name(metadata,rows,admission))!=
                    "coherent_down_k7_shared_rows_n8_split5",
            "coherent K7 down dispatch or fallback mismatch");
        require(std::string(shared.dispatch_name(metadata,rows,
                    Admission::ordinary))==
                    "coherent_down_k7_shared_rows_n8_split5" &&
                std::string(shared.dispatch_name(metadata,rows,
                    Admission::target_continuation_down))==
                    "coherent_down_k7_shared_rows_n8_split5",
            "coherent K7 down scalar and correction admission differ");
        cuda_check(cudaMemset(candidate,0xa5,
            (8u*out_features+guard)*sizeof(std::uint16_t)),
            "K7 down tail canary");
        old.forward(weights,metadata,input,control,rows,nullptr,admission);
        shared.forward(weights,metadata,input,candidate,rows,nullptr,admission);
        cuda_check(cudaDeviceSynchronize(),"K7 down differential synchronize");
        std::vector<std::uint16_t> reference(rows*out_features),actual(rows*out_features);
        cuda_check(cudaMemcpy(reference.data(),control,reference.size()*sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),"K7 down control download");
        cuda_check(cudaMemcpy(actual.data(),candidate,actual.size()*sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),"K7 down shared download");
        const auto other_admission = admission == Admission::ordinary ?
            Admission::target_continuation_down : Admission::ordinary;
        shared.forward(weights,metadata,input,control,rows,nullptr,other_admission);
        std::vector<std::uint16_t> other(actual.size());
        cuda_check(cudaMemcpy(other.data(),control,other.size()*sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
            "coherent K7 down alternate-admission download");
        require(other==actual,
            "coherent K7 down scalar/correction arithmetic differs");
        std::vector<std::uint16_t> tail(8u*out_features+guard-actual.size());
        cuda_check(cudaMemcpy(tail.data(),candidate+actual.size(),
            tail.size()*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
            "K7 down tail download");
        require(std::all_of(tail.begin(),tail.end(),
            [](std::uint16_t bits){return bits==0xa5a5u;}),
            "coherent K7 down wrote inactive row or tail");
        double error_sq=0.0,norm_sq=0.0,max_abs=0.0;
        for(int row=0;row<rows;++row) {
            const auto first=actual.begin()+static_cast<std::size_t>(row)*out_features;
            auto& snapshot=row_snapshots[row];
            if(snapshot.empty())snapshot.assign(first,first+out_features);
            else require(std::equal(snapshot.begin(),snapshot.end(),first),
                "coherent K7 down row changed with verifier width");
            for(int j=0;j<coherent_down_witness::width;++j) {
                const int col=coherent_down_witness::group*
                    coherent_down_witness::width+j;
                const double value=coherent_down_witness::from_half(
                    actual[static_cast<std::size_t>(row)*out_features+col]);
                const double expected=oracle[row].fp64_output[j];
                require(std::isfinite(value)&&std::isfinite(expected),
                    "coherent K7 down independent FP64 nonfinite");
                const double error=value-expected;
                error_sq+=error*error;norm_sq+=expected*expected;
                max_abs=std::max(max_abs,std::abs(error));
            }
        }
        const auto rel_l2=std::sqrt(error_sq/std::max(norm_sq,1e-30));
        const auto metrics=compare(actual,reference);
        require(std::isfinite(metrics.max_abs) &&
                std::isfinite(metrics.relative_l2),
            "coherent K7 down or old-policy output nonfinite");
        require(max_abs<=0.02 && rel_l2<=0.002,
            "coherent K7 down exceeds independent FP64 operator bound");
        std::cout<<"COHERENT_DOWN_K7 M="<<rows
                 <<" oracle_max_abs="<<max_abs
                 <<" oracle_relative_l2="<<rel_l2
                 <<" control_max_abs="<<metrics.max_abs
                 <<" resident_capacity="<<shared.coherent_down_k7_resident_capacity_for_test()
                 <<" shared_bytes=12800\n";
    }
    std::vector<std::uint16_t> scalar(out_features),correction(out_features);
    shared.forward(weights,metadata,input,candidate,1,nullptr,Admission::ordinary);
    cuda_check(cudaMemcpy(scalar.data(),candidate,scalar.size()*sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),"K7 down scalar download");
    shared.forward(weights,metadata,input,candidate,1,nullptr,
                   Admission::target_continuation_down);
    cuda_check(cudaMemcpy(correction.data(),candidate,
        correction.size()*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
        "K7 down correction download");
    require(scalar==correction,"coherent K7 down scalar/correction M1 mismatch");
    require(Exl3CudaLinearWorkspace::process_coherent_down_k7_calls_for_test()-
                before_calls==14 &&
            Exl3CudaLinearWorkspace::process_coherent_down_k7_rows_for_test()-
                before_rows==52,
        "coherent K7 down dispatch counts or rows mismatch");
    std::cout<<"COHERENT_DOWN_K7_OPERATOR PASS calls=14 rows=52\n";
}

void run_coherent_wide_k6_operator(const std::filesystem::path& model_dir) {
    using Admission=ninfer::exl3::Exl3CudaLinearAdmission;
    const bool split10=std::getenv("NINFER_EXL3_TEST_COHERENT_WIDE_K6_SPLIT10") &&
        std::strcmp(std::getenv("NINFER_EXL3_TEST_COHERENT_WIDE_K6_SPLIT10"),"1")==0;
    struct Shape {
        const char* name; const char* prefix;
        int in_features,out_features,operation;
        Admission continuation;
        bool gate_up,o,z,qkv,q;
        const char* dispatch;
    };
    const Shape shapes[] = {
        {"q","model.language_model.layers.19.self_attn.q_proj.",5120,12288,0,
            Admission::target_continuation_q,false,false,false,false,true,
            "coherent_wide_k6_q_n8_m16_split5"},
        {"qkv","model.language_model.layers.4.linear_attn.in_proj_qkv.",5120,10240,1,
            Admission::target_continuation_qkv,false,false,false,true,false,
            "coherent_wide_k6_qkv_n8_m16_split5"},
        {"z","model.language_model.layers.5.linear_attn.in_proj_z.",5120,6144,2,
            Admission::target_continuation_z,false,false,true,false,false,
            "coherent_wide_k6_z_n8_m16_split5"},
        {"o","model.language_model.layers.19.self_attn.o_proj.",6144,5120,3,
            Admission::target_continuation_o,false,true,false,false,false,
            "coherent_wide_k6_o_n8_m16_split5"},
        {"gate","model.language_model.layers.4.mlp.gate_proj.",5120,17408,4,
            Admission::target_continuation_gate_up,true,false,false,false,false,
            "coherent_wide_k6_gate_up_n8_m16_split5"},
        {"up","model.language_model.layers.19.mlp.up_proj.",5120,17408,4,
            Admission::target_continuation_gate_up,true,false,false,false,false,
            "coherent_wide_k6_gate_up_n8_m16_split5"}};
    const auto collection=ninfer::exl3::inspect_indexed_directory(model_dir);
    constexpr std::size_t guard=64;
    for(const auto& shape:shapes) {
        std::vector<TensorPayload> host;
        for(const char* suffix:{"trellis","suh","svh","mul1"})
            host.push_back(load_indexed_tensor(collection,
                (std::string(shape.prefix)+suffix).c_str()));
        require(host[0].bytes().size()==
                    static_cast<std::size_t>(shape.in_features/16)*
                    (shape.out_features/16)*96u*sizeof(std::uint16_t) &&
                host[1].bytes().size()==
                    static_cast<std::size_t>(shape.in_features)*sizeof(std::uint16_t) &&
                host[2].bytes().size()==
                    static_cast<std::size_t>(shape.out_features)*sizeof(std::uint16_t) &&
                host[3].bytes().size()==sizeof(std::int32_t),
            std::string("coherent wide K6 real weight extent: ")+shape.name);
        DeviceBuffer d_trellis,d_suh,d_svh,d_mul1,d_input,d_scalar,d_verifier;
        const auto upload=[](DeviceBuffer& owner,const TensorPayload& tensor,
                             const char* label) {
            auto* ptr=allocate_device<std::byte>(owner,tensor.bytes().size(),label);
            cuda_check(cudaMemcpy(ptr,tensor.bytes().data(),tensor.bytes().size(),
                cudaMemcpyHostToDevice),label);
            return ptr;
        };
        const Exl3CudaLinearWeights weights{
            reinterpret_cast<const std::uint16_t*>(upload(d_trellis,host[0],"wide trellis")),
            reinterpret_cast<const std::uint16_t*>(upload(d_suh,host[1],"wide suh")),
            reinterpret_cast<const std::uint16_t*>(upload(d_svh,host[2],"wide svh")),
            reinterpret_cast<const std::int32_t*>(upload(d_mul1,host[3],"wide mul1"))};
        std::vector<std::uint16_t> input_bits(8u*shape.in_features);
        for(std::size_t i=0;i<input_bits.size();++i) {
            const auto magnitude=static_cast<std::uint16_t>(
                0x2800u+((i*73u+i/shape.in_features*31u)%0x800u));
            input_bits[i]=static_cast<std::uint16_t>(
                magnitude|(((i/19u)&1u)?0x8000u:0u));
        }
        auto* input=allocate_device<std::uint16_t>(d_input,input_bits.size(),"wide input");
        cuda_check(cudaMemcpy(input,input_bits.data(),
            input_bits.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice),
            "wide input upload");
        const std::size_t extent=8u*shape.out_features+guard;
        auto* scalar=allocate_device<std::uint16_t>(d_scalar,extent,"wide scalar");
        auto* verifier=allocate_device<std::uint16_t>(d_verifier,extent,"wide verifier");
        const Exl3CudaLinearMetadata metadata{
            shape.in_features,shape.out_features,6,false,true,false};
        _putenv_s("NINFER_EXL3_TARGET_COHERENT_WIDE_K6","0");
        Exl3CudaLinearWorkspace old(shape.in_features,shape.out_features,8,
            false,shape.gate_up,false,shape.o,shape.z,false,false,
            {},{},shape.qkv,false,shape.q);
        _putenv_s("NINFER_EXL3_TARGET_COHERENT_WIDE_K6","1");
        _putenv_s("NINFER_EXL3_TARGET_COHERENT_WIDE_K6_SPLIT10",split10?"1":"0");
        Exl3CudaLinearWorkspace candidate(shape.in_features,shape.out_features,
            split10?16:8,
            false,shape.gate_up,false,shape.o,shape.z,false,false,
            {},{},shape.qkv,false,shape.q);
        std::string dispatch=shape.dispatch;
        if(split10)dispatch.replace(dispatch.size()-1,1,"10");
        require(candidate.coherent_wide_k6_operation_for_test()==shape.operation &&
                candidate.coherent_wide_k6_resident_capacity_for_test()>0 &&
                candidate.coherent_wide_k6_registers_per_thread_for_test()>0 &&
                candidate.coherent_wide_k6_shared_bytes_for_test()==12288,
            std::string("coherent wide K6 resource/owner gate: ")+shape.name);
        require(std::string(candidate.dispatch_name(
                Exl3CudaLinearMetadata{shape.in_features,shape.out_features,7,
                    false,true,false},1,Admission::ordinary))!=dispatch &&
                std::string(candidate.dispatch_name(metadata,9,
                    Admission::ordinary))!=dispatch &&
                std::string(candidate.dispatch_name(metadata,1,
                    Admission::target_continuation_down))!=dispatch,
            std::string("coherent wide K6 metadata/row escape: ")+shape.name);
        const auto before_calls=Exl3CudaLinearWorkspace::
            coherent_wide_k6_calls_for_test(shape.operation);
        const auto before_rows=Exl3CudaLinearWorkspace::
            coherent_wide_k6_rows_for_test(shape.operation);
        const auto before_split10=Exl3CudaLinearWorkspace::
            coherent_wide_k6_split10_calls_for_test(shape.operation);
        std::array<coherent_wide_k6_witness::Output,8> oracle;
        for(int row=0;row<8;++row)
            oracle[row]=coherent_wide_k6_witness::evaluate(
                input_bits,host[1].typed<std::uint16_t>("F16"),
                host[2].typed<std::uint16_t>("F16"),
                host[0].typed<std::uint16_t>("I16"),
                static_cast<std::uint32_t>(host[3].typed<std::int32_t>("I32")[0]),
                row,shape.in_features,shape.out_features);
        std::array<std::vector<std::uint16_t>,8> row_snapshots;
        for(int rows:{1,2,3,4,7,8}) {
            require(std::string(old.dispatch_name(metadata,rows,
                        Admission::ordinary))!=dispatch &&
                    std::string(candidate.dispatch_name(metadata,rows,
                        Admission::ordinary))==dispatch &&
                    std::string(candidate.dispatch_name(metadata,rows,
                        shape.continuation))==dispatch,
                std::string("coherent wide K6 dispatch: ")+shape.name);
            cuda_check(cudaMemset(scalar,0xa5,extent*sizeof(std::uint16_t)),
                "wide scalar canary initialize");
            cuda_check(cudaMemset(verifier,0xa5,extent*sizeof(std::uint16_t)),
                "wide verifier canary initialize");
            candidate.forward(weights,metadata,input,scalar,rows,nullptr,
                Admission::ordinary);
            candidate.forward(weights,metadata,input,verifier,rows,nullptr,
                shape.continuation);
            std::vector<std::uint16_t> actual(extent),other(extent);
            cuda_check(cudaMemcpy(actual.data(),scalar,extent*sizeof(std::uint16_t),
                cudaMemcpyDeviceToHost),"wide scalar download");
            cuda_check(cudaMemcpy(other.data(),verifier,extent*sizeof(std::uint16_t),
                cudaMemcpyDeviceToHost),"wide verifier download");
            require(actual==other,
                std::string("coherent wide K6 scalar/verifier bits: ")+shape.name);
            const std::size_t active=static_cast<std::size_t>(rows)*shape.out_features;
            require(std::all_of(actual.begin()+active,actual.end(),
                [](std::uint16_t bits){return bits==0xa5a5u;}),
                std::string("coherent wide K6 inactive tail: ")+shape.name);
            double error_sq=0.0,norm_sq=0.0,max_abs=0.0;
            for(int row=0;row<rows;++row) {
                const auto first=actual.begin()+
                    static_cast<std::size_t>(row)*shape.out_features;
                auto& snapshot=row_snapshots[row];
                if(snapshot.empty())snapshot.assign(first,first+shape.out_features);
                else require(std::equal(snapshot.begin(),snapshot.end(),first),
                    std::string("coherent wide K6 row-width stability: ")+shape.name);
                for(int j=0;j<coherent_wide_k6_witness::width;++j) {
                    const int column=coherent_wide_k6_witness::group*
                        coherent_wide_k6_witness::width+j;
                    const double value=coherent_down_witness::from_half(
                        actual[static_cast<std::size_t>(row)*shape.out_features+column]);
                    const double reference=oracle[row][j];
                    require(std::isfinite(value)&&std::isfinite(reference),
                        std::string("coherent wide K6 oracle finite: ")+shape.name);
                    const double error=value-reference;
                    error_sq+=error*error;norm_sq+=reference*reference;
                    max_abs=std::max(max_abs,std::abs(error));
                }
            }
            const double relative_l2=std::sqrt(error_sq/std::max(norm_sq,1e-30));
            std::cout<<"COHERENT_WIDE_K6 op="<<shape.name<<" M="<<rows
                     <<" max_abs="<<max_abs<<" relative_l2="<<relative_l2
                     <<" resident_capacity="<<candidate.coherent_wide_k6_resident_capacity_for_test()
                     <<" registers="<<candidate.coherent_wide_k6_registers_per_thread_for_test()
                     <<" shared_bytes="<<candidate.coherent_wide_k6_shared_bytes_for_test()
                     <<" split_count="<<(split10?10:5)<<'\n';
            require(max_abs<=0.02 && relative_l2<=0.002,
                std::string("coherent wide K6 FP64 bound: ")+shape.name);
        }
        require(Exl3CudaLinearWorkspace::coherent_wide_k6_calls_for_test(
                    shape.operation)-before_calls==12 &&
                Exl3CudaLinearWorkspace::coherent_wide_k6_rows_for_test(
                    shape.operation)-before_rows==50 &&
                Exl3CudaLinearWorkspace::coherent_wide_k6_split10_calls_for_test(
                    shape.operation)-before_split10==(split10?12u:0u),
            std::string("coherent wide K6 actual calls/rows: ")+shape.name);
    }
}

} // namespace

int main() {
    try {
        if (env_or_empty("NINFER_EXL3_TEST_COHERENT_DOWN_CPU") == "1" ||
            env_or_empty("NINFER_EXL3_TEST_COHERENT_DOWN_K7_CPU") == "1") {
            const bool k7_cpu =
                env_or_empty("NINFER_EXL3_TEST_COHERENT_DOWN_K7_CPU") == "1";
            const auto model_text = env_or_empty("NINFER_EXL3_TARGET_PATH");
            require(!model_text.empty(), "coherent down CPU witness requires model path");
            const auto collection = ninfer::exl3::inspect_indexed_directory(model_text);
            std::vector<TensorPayload> host;
            for (const char* suffix : {"trellis", "suh", "svh", "mul1"})
                host.push_back(load_indexed_tensor(collection,
                    (std::string(k7_cpu ? kDownK7Prefix : kDownPrefix) + suffix).c_str()));
            std::vector<std::uint16_t> input_bits(8u * 17408u);
            for (std::size_t index = 0; index < input_bits.size(); ++index) {
                const std::uint16_t magnitude = static_cast<std::uint16_t>(
                    0x2800u + ((index * 73u + index / 17408u * 31u) % 0x800u));
                input_bits[index] = static_cast<std::uint16_t>(
                    magnitude | (((index / 19u) & 1u) ? 0x8000u : 0u));
            }
            const auto trellis_k7 = host[0].typed<std::uint16_t>("I16");
            if (k7_cpu) {
                require(trellis_k7.size() == static_cast<std::size_t>(17408 / 16) *
                            (5120 / 16) * 7u * 16u,
                        "coherent K7 down CPU trellis extent mismatch");
                for (int row : {0, 1, 2, 3, 6, 7}) {
                    const auto result = coherent_down_witness::evaluate_down_k7(
                        input_bits, host[1].typed<std::uint16_t>("F16"),
                        host[2].typed<std::uint16_t>("F16"), trellis_k7,
                        static_cast<std::uint32_t>(
                            host[3].typed<std::int32_t>("I32")[0]), row);
                    require(std::all_of(result.fp64_output.begin(),
                                result.fp64_output.end(),
                                [](double value) { return std::isfinite(value); }),
                            "coherent K7 down CPU witness nonfinite");
                    std::cout << "COHERENT_DOWN_K7_CPU row=" << row
                              << " group=" << coherent_down_witness::group
                              << " fp64_first=" << result.fp64_output[0] << '\n';
                }
                return 0;
            }
            for (int row : {0, 1}) {
                const auto result = coherent_down_witness::evaluate(
                    input_bits, host[1].typed<std::uint16_t>("F16"),
                    host[2].typed<std::uint16_t>("F16"),
                    host[0].typed<std::uint16_t>("I16"),
                    static_cast<std::uint32_t>(
                        host[3].typed<std::int32_t>("I32")[0]), row);
                int scalar_split_bits = 0, scalar_oracle_bits = 0;
                double scalar_oracle_max = 0.0;
                for (int j = 0; j < coherent_down_witness::width; ++j) {
                    scalar_split_bits += result.scalar_fp32_output[j] !=
                        result.split5_fp32_output[j];
                    const auto oracle_half = coherent_down_witness::to_half(
                        static_cast<float>(result.fp64_output[j]));
                    scalar_oracle_bits += result.scalar_fp32_output[j] != oracle_half;
                    scalar_oracle_max = std::max(scalar_oracle_max, std::abs(
                        static_cast<double>(coherent_down_witness::from_half(
                            result.scalar_fp32_output[j])) - result.fp64_output[j]));
                }
                require(std::isfinite(scalar_oracle_max) && scalar_oracle_bits > 0,
                        "coherent down witness did not expose represented cast boundary");
                std::cout << "COHERENT_DOWN_CPU row=" << row
                          << " group=" << coherent_down_witness::group
                          << " scalar_split_bits=" << scalar_split_bits
                          << " scalar_oracle_bits=" << scalar_oracle_bits
                          << " scalar_oracle_max_abs=" << scalar_oracle_max << '\n';
            }
            return 0;
        }
        int device_count = 0;
        const auto device_status = cudaGetDeviceCount(&device_count);
        if (device_status != cudaSuccess || device_count == 0) {
            std::cerr << "SKIP: CUDA device unavailable: "
                      << cudaGetErrorString(device_status) << '\n';
            return 77;
        }
        const auto model_text = env_or_empty("NINFER_EXL3_TARGET_PATH");
        if (env_or_empty("NINFER_EXL3_TEST_FAST_SMALL_M_DOWN") == "1" ||
            env_or_empty("NINFER_EXL3_TEST_COHERENT_DOWN_K6") == "1") {
            require(!model_text.empty(), "small-M down test requires model path");
            run_fast_small_m_down_operator(std::filesystem::path(model_text));
            return 0;
        }
        if (env_or_empty("NINFER_EXL3_TEST_COHERENT_WIDE_K6") == "1") {
            require(!model_text.empty(),"coherent wide K6 test requires model path");
            run_coherent_wide_k6_operator(std::filesystem::path(model_text));
            return 0;
        }
        if (env_or_empty("NINFER_EXL3_TEST_COHERENT_DOWN_K7") == "1") {
            require(!model_text.empty(),"coherent K7 down test requires model path");
            run_coherent_down_k7_operator(std::filesystem::path(model_text));
            return 0;
        }
        if (env_or_empty("NINFER_EXL3_TEST_COHERENT_O_K7") == "1") {
            require(!model_text.empty(),"coherent K7 O test requires model path");
            run_coherent_o_k7_operator(std::filesystem::path(model_text));
            return 0;
        }
        const auto oracle_text = env_or_empty("NINFER_EXL3_ORACLE_PATH");
        if (model_text.empty() || oracle_text.empty()) {
            std::cerr << "SKIP: set NINFER_EXL3_TARGET_PATH and NINFER_EXL3_ORACLE_PATH\n";
            return 77;
        }
        const std::filesystem::path model_dir(model_text);
        const std::filesystem::path oracle_path(oracle_text);
        require(std::filesystem::exists(model_dir), "model directory does not exist");
        require(std::filesystem::exists(oracle_path), "oracle fixture does not exist");

        const auto collection = ninfer::exl3::inspect_indexed_directory(model_dir);
        std::vector<TensorPayload> keep_alive;
        keep_alive.reserve(4);
        keep_alive.push_back(load_indexed_tensor(collection, kTrellisName));
        keep_alive.push_back(load_indexed_tensor(collection, kSuhName));
        keep_alive.push_back(load_indexed_tensor(collection, kSvhName));
        keep_alive.push_back(load_indexed_tensor(collection, kMul1Name));
        const Exl3LinearTensors host_tensors{
            &keep_alive[0], &keep_alive[1], &keep_alive[2], &keep_alive[3]};
        const Exl3LinearMetadata host_metadata{5120, 17408, 5, false, true, false, "mul1"};
        ninfer::exl3::validate_linear_tensors(host_tensors, host_metadata);

        const auto oracle_header = ninfer::exl3::inspect_file(oracle_path);
        const auto canonical_input =
            ninfer::exl3::read_tensor(oracle_path, oracle_header, "linear_input");
        const auto canonical_output =
            ninfer::exl3::read_tensor(oracle_path, oracle_header, "linear_output");
        const auto canonical_input_bits = canonical_input.typed<std::uint16_t>("F16");
        const auto canonical_output_bits = canonical_output.typed<std::uint16_t>("F16");
        require(canonical_input.info.shape == std::vector<std::uint64_t>{1, 14, 5120},
                "canonical input shape mismatch");
        require(canonical_output.info.shape == std::vector<std::uint64_t>{1, 14, 17408},
                "canonical output shape mismatch");

        DeviceBuffer d_trellis;
        DeviceBuffer d_suh;
        DeviceBuffer d_svh;
        DeviceBuffer d_mul1;
        DeviceBuffer d_input;
        DeviceBuffer d_output;
        auto* trellis = allocate_device<std::uint16_t>(d_trellis, 320u * 1088u * 80u,
                                                        "cudaMalloc trellis");
        auto* suh = allocate_device<std::uint16_t>(d_suh, 5120, "cudaMalloc suh");
        auto* svh = allocate_device<std::uint16_t>(d_svh, 17408, "cudaMalloc svh");
        auto* mul1 = allocate_device<std::int32_t>(d_mul1, 1, "cudaMalloc mul1");
        auto* input = allocate_device<std::uint16_t>(d_input, 16u * 5120u,
                                                      "cudaMalloc input");
        auto* output = allocate_device<std::uint16_t>(d_output, 16u * 17408u,
                                                       "cudaMalloc output");
        cuda_check(cudaMemcpy(trellis, keep_alive[0].bytes().data(), keep_alive[0].bytes().size(),
                              cudaMemcpyHostToDevice), "copy trellis");
        cuda_check(cudaMemcpy(suh, keep_alive[1].bytes().data(), keep_alive[1].bytes().size(),
                              cudaMemcpyHostToDevice), "copy suh");
        cuda_check(cudaMemcpy(svh, keep_alive[2].bytes().data(), keep_alive[2].bytes().size(),
                              cudaMemcpyHostToDevice), "copy svh");
        cuda_check(cudaMemcpy(mul1, keep_alive[3].bytes().data(), keep_alive[3].bytes().size(),
                              cudaMemcpyHostToDevice), "copy mul1");

        const Exl3CudaLinearWeights device_weights{trellis, suh, svh, mul1};
        const Exl3CudaLinearMetadata device_metadata{5120, 17408, 5, false, true, false};
        Exl3CudaLinearWorkspace workspace(16);
        std::cout << "workspace_bytes=" << workspace.workspace_bytes() << '\n';
        std::cout << "resident_weight_bytes="
                  << (keep_alive[0].bytes().size() + keep_alive[1].bytes().size() +
                      keep_alive[2].bytes().size() + keep_alive[3].bytes().size()) << '\n';

        run_negative_tests(workspace, device_weights, device_metadata, input, output);

        const auto reference_14 =
            ninfer::exl3::exl3_linear_reference(canonical_input, host_tensors, host_metadata);
        const auto reference_14_metrics = compare(reference_14.fp16_bits, canonical_output_bits);
        print_metrics("E2 reference vs E1 canonical M=14", reference_14_metrics);
        require(reference_14_metrics.max_abs <= 0.0015 &&
                    reference_14_metrics.relative_l2 <= 0.0007,
                "E2 reference no longer agrees with canonical oracle");

        std::vector<int> row_counts{1, 2, 4, 8, 14, 16};
        if (const auto only_rows = env_or_empty("NINFER_EXL3_ONLY_M"); !only_rows.empty()) {
            row_counts = {std::stoi(only_rows)};
        }
        for (const int rows : row_counts) {
            const auto input_payload = make_input(canonical_input, rows);
            const auto input_bits = input_payload.typed<std::uint16_t>("F16");
            cuda_check(cudaMemcpy(input, input_bits.data(), input_bits.size_bytes(),
                                  cudaMemcpyHostToDevice), "copy EXL3 input");
            workspace.forward(device_weights, device_metadata, input, output, rows);
            cuda_check(cudaDeviceSynchronize(), "synchronize EXL3 correctness result");
            std::vector<std::uint16_t> actual(input_bits.size() / 5120u * 17408u);
            cuda_check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost), "copy EXL3 output");

            auto expected = expected_rows(canonical_output_bits, rows);
            const auto metrics = compare(actual, expected);
            print_metrics((std::string("CUDA vs E1 row fixture M=") + std::to_string(rows)).c_str(),
                          metrics);
            require(metrics.max_abs <= 0.0015 && metrics.relative_l2 <= 0.0007,
                    "optimized EXL3 result failed E1/E2 tolerance");
            if (rows == 16) {
                const auto reference_16 =
                    ninfer::exl3::exl3_linear_reference(input_payload, host_tensors, host_metadata);
                const auto reference_metrics = compare(actual, reference_16.fp16_bits);
                print_metrics("CUDA vs E2 reference M=16", reference_metrics);
                require(reference_metrics.max_abs <= 0.0015 &&
                            reference_metrics.relative_l2 <= 0.0007,
                        "optimized EXL3 result failed E2 reference tolerance");
            }

            const double cuda_us = benchmark(workspace, device_weights, device_metadata,
                                              input, output, rows);
            std::cout << "benchmark M=" << rows << " cuda_us=" << std::setprecision(10)
                      << cuda_us << " kernel_launches=3\n";
        }
        std::cout << "ninfer_exl3_cuda_linear_test: PASS\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "FAIL: " << exception.what() << '\n';
        return 1;
    }
}
