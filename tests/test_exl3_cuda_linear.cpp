#include "exl3/linear_cuda.h"
#include "exl3/linear_reference.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
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

} // namespace

int main() {
    try {
        int device_count = 0;
        const auto device_status = cudaGetDeviceCount(&device_count);
        if (device_status != cudaSuccess || device_count == 0) {
            std::cerr << "SKIP: CUDA device unavailable: "
                      << cudaGetErrorString(device_status) << '\n';
            return 77;
        }
        const auto model_text = env_or_empty("NINFER_EXL3_TARGET_PATH");
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
