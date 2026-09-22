#include "core/device.h"
#include "ops/kv_cache/oscar_int2_g128.h"
#include "ops/softmax_attention/oscar_history/launch.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kQHeads = 24;
constexpr int kKVHeads = 4;
constexpr int kGqa = 6;
constexpr int kHeadDim = 256;
constexpr int kGroups = 2;
constexpr int kGroupSize = 128;
constexpr int kCodeBytes = 64;
constexpr float kAttentionScale = 1.0F / 16.0F;
constexpr float kParityTolerance = 1.0e-4F;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

std::uint32_t mix(std::uint32_t value) noexcept {
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    value ^= value >> 16U;
    return value;
}

float fixture_value(std::uint32_t stream, int token, int kv_head, int dimension) noexcept {
    const std::uint32_t seed = mix(0xD42A0001U ^ stream * 0x9e3779b9U ^
                                   static_cast<std::uint32_t>(token + 11) * 0x85ebca6bU ^
                                   static_cast<std::uint32_t>(kv_head + 5) * 0xc2b2ae35U ^
                                   static_cast<std::uint32_t>(dimension + 7) * 0x27d4eb2fU);
    const float centered = static_cast<float>(static_cast<std::int32_t>(seed % 8192U) - 4096);
    const float wave = 0.75F * std::sin(static_cast<float>(dimension + 1) * 0.017F +
                                        static_cast<float>(token) * 0.003F);
    return centered / 4096.0F + wave + static_cast<float>(kv_head) * 0.07F;
}

struct Fixture {
    int history = 0;
    std::vector<float> q;
    std::vector<std::uint8_t> k_packed;
    std::vector<float> k_metadata;
    std::vector<std::uint8_t> v_packed;
    std::vector<float> v_metadata;
};

Fixture make_fixture(int history) {
    Fixture fixture;
    fixture.history = history;
    fixture.q.resize(static_cast<std::size_t>(kQHeads) * kHeadDim);
    fixture.k_packed.resize(static_cast<std::size_t>(history) * kKVHeads * kCodeBytes);
    fixture.k_metadata.resize(static_cast<std::size_t>(history) * kKVHeads * kGroups * 2);
    fixture.v_packed.resize(static_cast<std::size_t>(history) * kKVHeads * kCodeBytes);
    fixture.v_metadata.resize(static_cast<std::size_t>(history) * kKVHeads * kGroups * 2);

    for (int query_head = 0; query_head < kQHeads; ++query_head) {
        for (int dimension = 0; dimension < kHeadDim; ++dimension) {
            fixture.q[static_cast<std::size_t>(query_head) * kHeadDim + dimension] =
                0.6F * std::sin(static_cast<float>(query_head + 1) * 0.23F +
                                 static_cast<float>(dimension + 1) * 0.013F) +
                0.25F * std::cos(static_cast<float>(dimension + 3) * 0.031F);
        }
    }

    std::vector<float> row(kHeadDim);
    for (int token = 0; token < history; ++token) {
        for (int kv_head = 0; kv_head < kKVHeads; ++kv_head) {
            for (int dimension = 0; dimension < kHeadDim; ++dimension) {
                row[dimension] = fixture_value(0x4B4B4B4BU, token, kv_head, dimension);
            }
            const auto encoded_k = ninfer::ops::oscar_int2_g128_encode(row.data(), kHeadDim, 0.96F);
            const std::size_t row_index = static_cast<std::size_t>(token) * kKVHeads + kv_head;
            std::copy(encoded_k.packed.begin(), encoded_k.packed.end(),
                      fixture.k_packed.begin() + row_index * kCodeBytes);
            std::copy(encoded_k.scales_zeros.begin(), encoded_k.scales_zeros.end(),
                      fixture.k_metadata.begin() + row_index * kGroups * 2);

            for (int dimension = 0; dimension < kHeadDim; ++dimension) {
                row[dimension] = fixture_value(0x56565656U, token, kv_head, dimension);
            }
            const auto encoded_v = ninfer::ops::oscar_int2_g128_encode(row.data(), kHeadDim, 0.92F);
            std::copy(encoded_v.packed.begin(), encoded_v.packed.end(),
                      fixture.v_packed.begin() + row_index * kCodeBytes);
            std::copy(encoded_v.scales_zeros.begin(), encoded_v.scales_zeros.end(),
                      fixture.v_metadata.begin() + row_index * kGroups * 2);
        }
    }
    return fixture;
}

float decode_host(const std::uint8_t* packed, const float* metadata, int dimension) noexcept {
    const int byte = dimension & 63;
    const int shift = (dimension >> 6) << 1;
    const std::uint8_t symbol = static_cast<std::uint8_t>((packed[byte] >> shift) & 3U);
    const int group = dimension >> 7;
    return (static_cast<float>(symbol) - metadata[group * 2 + 1]) * metadata[group * 2];
}

struct ReferenceOutput {
    std::vector<float> scores;
    std::vector<float> softmax;
    std::vector<float> output;
};

ReferenceOutput reference(const Fixture& fixture) {
    const int history = fixture.history;
    ReferenceOutput result;
    result.scores.resize(static_cast<std::size_t>(kQHeads) * history);
    result.softmax.resize(result.scores.size());
    result.output.resize(static_cast<std::size_t>(kQHeads) * kHeadDim);

    for (int query_head = 0; query_head < kQHeads; ++query_head) {
        const int kv_head = query_head / kGqa;
        for (int token = 0; token < history; ++token) {
            const std::size_t row = static_cast<std::size_t>(token) * kKVHeads + kv_head;
            const auto* packed = fixture.k_packed.data() + row * kCodeBytes;
            const auto* metadata = fixture.k_metadata.data() + row * kGroups * 2;
            float dot = 0.0F;
            for (int dimension = 0; dimension < kHeadDim; ++dimension) {
                dot += fixture.q[static_cast<std::size_t>(query_head) * kHeadDim + dimension] *
                       decode_host(packed, metadata, dimension);
            }
            result.scores[static_cast<std::size_t>(query_head) * history + token] =
                dot * kAttentionScale;
        }

        const float* score_row = result.scores.data() + static_cast<std::size_t>(query_head) * history;
        float maximum = -std::numeric_limits<float>::infinity();
        for (int token = 0; token < history; ++token) maximum = std::max(maximum, score_row[token]);
        float sum = 0.0F;
        for (int token = 0; token < history; ++token) {
            const float value = std::exp(score_row[token] - maximum);
            result.softmax[static_cast<std::size_t>(query_head) * history + token] = value;
            sum += value;
        }
        for (int token = 0; token < history; ++token) {
            result.softmax[static_cast<std::size_t>(query_head) * history + token] /= sum;
        }
    }

    for (int query_head = 0; query_head < kQHeads; ++query_head) {
        const int kv_head = query_head / kGqa;
        for (int dimension = 0; dimension < kHeadDim; ++dimension) {
            float value = 0.0F;
            for (int token = 0; token < history; ++token) {
                const std::size_t row = static_cast<std::size_t>(token) * kKVHeads + kv_head;
                const float decoded = decode_host(
                    fixture.v_packed.data() + row * kCodeBytes,
                    fixture.v_metadata.data() + row * kGroups * 2, dimension);
                value += result.softmax[static_cast<std::size_t>(query_head) * history + token] * decoded;
            }
            result.output[static_cast<std::size_t>(query_head) * kHeadDim + dimension] = value;
        }
    }
    return result;
}

struct Metrics {
    float max_abs = 0.0F;
    float mean_abs = 0.0F;
    float relative_l2 = 0.0F;
};

Metrics compare(const std::vector<float>& reference_values, const std::vector<float>& actual) {
    require(reference_values.size() == actual.size(), "comparison size mismatch");
    double sum_abs = 0.0;
    double sum_diff_sq = 0.0;
    double sum_ref_sq = 0.0;
    float max_abs = 0.0F;
    for (std::size_t i = 0; i < reference_values.size(); ++i) {
        require(std::isfinite(actual[i]), "GPU result contains NaN or Inf");
        const float difference = actual[i] - reference_values[i];
        const float absolute = std::fabs(difference);
        max_abs = std::max(max_abs, absolute);
        sum_abs += absolute;
        sum_diff_sq += static_cast<double>(difference) * difference;
        sum_ref_sq += static_cast<double>(reference_values[i]) * reference_values[i];
    }
    Metrics result;
    result.max_abs = max_abs;
    result.mean_abs = static_cast<float>(sum_abs / static_cast<double>(reference_values.size()));
    result.relative_l2 = static_cast<float>(std::sqrt(sum_diff_sq / std::max(sum_ref_sq, 1.0e-30)));
    return result;
}

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;
    std::size_t count = 0;

    explicit DeviceBuffer(std::size_t elements) : count(elements) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (data != nullptr) cudaFree(data);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct DeviceFixture {
    DeviceBuffer<float> q;
    DeviceBuffer<std::uint8_t> k_packed;
    DeviceBuffer<float> k_metadata;
    DeviceBuffer<std::uint8_t> v_packed;
    DeviceBuffer<float> v_metadata;
    DeviceBuffer<float> scores;
    DeviceBuffer<float> softmax;
    DeviceBuffer<float> output;

    explicit DeviceFixture(const Fixture& fixture)
        : q(fixture.q.size()),
          k_packed(fixture.k_packed.size()),
          k_metadata(fixture.k_metadata.size()),
          v_packed(fixture.v_packed.size()),
          v_metadata(fixture.v_metadata.size()),
          scores(static_cast<std::size_t>(kQHeads) * fixture.history),
          softmax(static_cast<std::size_t>(kQHeads) * fixture.history),
          output(static_cast<std::size_t>(kQHeads) * kHeadDim) {
        CUDA_CHECK(cudaMemcpy(q.data, fixture.q.data(), fixture.q.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(k_packed.data, fixture.k_packed.data(),
                              fixture.k_packed.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(k_metadata.data, fixture.k_metadata.data(),
                              fixture.k_metadata.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(v_packed.data, fixture.v_packed.data(),
                              fixture.v_packed.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(v_metadata.data, fixture.v_metadata.data(),
                              fixture.v_metadata.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
};

void launch(const Fixture& fixture, DeviceFixture& device) {
    ninfer::ops::detail::oscar_int2_g128_history_attention_launch(
        device.q.data, device.k_packed.data, device.k_metadata.data, device.v_packed.data,
        device.v_metadata.data, fixture.history, kAttentionScale, device.scores.data,
        device.softmax.data, device.output.data, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
}

struct DeviceOutput {
    std::vector<float> scores;
    std::vector<float> softmax;
    std::vector<float> output;
};

DeviceOutput copy_output(const Fixture& fixture, const DeviceFixture& device) {
    DeviceOutput result;
    result.scores.resize(static_cast<std::size_t>(kQHeads) * fixture.history);
    result.softmax.resize(result.scores.size());
    result.output.resize(static_cast<std::size_t>(kQHeads) * kHeadDim);
    CUDA_CHECK(cudaMemcpy(result.scores.data(), device.scores.data,
                          result.scores.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.softmax.data(), device.softmax.data,
                          result.softmax.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.output.data(), device.output.data,
                          result.output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

double benchmark_gpu(const Fixture& fixture, DeviceFixture& device, int repetitions) {
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    launch(fixture, device);
    CUDA_CHECK(cudaEventRecord(begin));
    for (int iteration = 0; iteration < repetitions; ++iteration) {
        ninfer::ops::detail::oscar_int2_g128_history_attention_launch(
            device.q.data, device.k_packed.data, device.k_metadata.data, device.v_packed.data,
            device.v_metadata.data, fixture.history, kAttentionScale, device.scores.data,
            device.softmax.data, device.output.data, nullptr);
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, begin, end));
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    return static_cast<double>(elapsed_ms) / repetitions;
}

double benchmark_cpu(const Fixture& fixture, int repetitions) {
    volatile float checksum = 0.0F;
    const auto begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < repetitions; ++iteration) {
        const ReferenceOutput value = reference(fixture);
        checksum += value.output[iteration % value.output.size()];
    }
    const auto end = std::chrono::steady_clock::now();
    require(std::isfinite(checksum), "CPU benchmark checksum is not finite");
    return std::chrono::duration<double, std::milli>(end - begin).count() / repetitions;
}

int benchmark_repetitions(int history) {
    if (history <= 512) return 30;
    if (history <= 4096) return 10;
    if (history <= 16384) return 5;
    return 3;
}

} // namespace

int main() {
    try {
        int device_count = 0;
        const cudaError_t count_error = cudaGetDeviceCount(&device_count);
        if (cuda_unavailable(count_error)) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        if (count_error != cudaSuccess) {
            std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_error) << '\n';
            return 1;
        }
        if (device_count == 0) {
            std::cout << "SKIP: no CUDA devices\n";
            return 77;
        }
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        require(properties.major == 12, "D4.2a requires the SM120a test target");

        const auto resources = ninfer::ops::detail::oscar_int2_g128_history_kernel_resources();
        std::cout << "Oscar historical INT2 fused kernel: GPU=" << properties.name
                  << " compute=" << properties.major << '.' << properties.minor
                  << " q_heads=24 kv_heads=4 gqa=6 D=256 group=128\n";
        std::cout << "resources score_regs=" << resources.score_registers
                  << " softmax_regs=" << resources.softmax_registers
                  << " av_regs=" << resources.av_registers
                  << " score_smem=" << resources.score_static_shared_bytes
                  << " softmax_smem=" << resources.softmax_static_shared_bytes
                  << " av_smem=" << resources.av_static_shared_bytes << "\n";
        std::cout << "history,gpu_ms,cpu_scalar_ms,speedup,gpu_us_per_history_token,effective_GBps"
                     ",score_max_abs,score_rel_l2,softmax_max_abs,softmax_rel_l2,av_max_abs,av_rel_l2\n";

        const std::vector<int> correctness_histories = {128, 512, 2048, 4096};
        const std::vector<int> benchmark_histories = {128, 512, 2048, 4096, 8192, 16384, 32768};
        for (const int history : benchmark_histories) {
            const Fixture fixture = make_fixture(history);
            DeviceFixture device(fixture);
            launch(fixture, device);
            const DeviceOutput actual = copy_output(fixture, device);
            if (std::find(correctness_histories.begin(), correctness_histories.end(), history) !=
                correctness_histories.end()) {
                const ReferenceOutput expected = reference(fixture);
                const Metrics score_metrics = compare(expected.scores, actual.scores);
                const Metrics softmax_metrics = compare(expected.softmax, actual.softmax);
                const Metrics av_metrics = compare(expected.output, actual.output);
                require(score_metrics.relative_l2 <= kParityTolerance,
                        "historical fused score parity exceeded tolerance at history " +
                            std::to_string(history));
                require(softmax_metrics.relative_l2 <= kParityTolerance,
                        "historical fused softmax parity exceeded tolerance at history " +
                            std::to_string(history));
                require(av_metrics.relative_l2 <= kParityTolerance,
                        "historical fused AV parity exceeded tolerance at history " +
                            std::to_string(history));
                std::cout << std::scientific << std::setprecision(9)
                          << "correctness history=" << history
                          << " score(max/rel)=" << score_metrics.max_abs << '/' << score_metrics.relative_l2
                          << " softmax(max/rel)=" << softmax_metrics.max_abs << '/'
                          << softmax_metrics.relative_l2 << " av(max/rel)=" << av_metrics.max_abs << '/'
                          << av_metrics.relative_l2 << " PASS\n" << std::defaultfloat;
            }

            const int gpu_repetitions = benchmark_repetitions(history);
            const int cpu_repetitions = history <= 4096 ? 2 : 1;
            const double gpu_ms = benchmark_gpu(fixture, device, gpu_repetitions);
            const double cpu_ms = benchmark_cpu(fixture, cpu_repetitions);
            const double speedup = cpu_ms / gpu_ms;
            const double bytes = 1216.0 * history + 49152.0;
            const double bandwidth = bytes / (gpu_ms * 1.0e6);
            std::cout << std::fixed << std::setprecision(6)
                      << history << ',' << gpu_ms << ',' << cpu_ms << ',' << speedup << ','
                      << (gpu_ms * 1000.0 / history) << ',' << bandwidth;
            if (std::find(correctness_histories.begin(), correctness_histories.end(), history) !=
                correctness_histories.end()) {
                const ReferenceOutput expected = reference(fixture);
                const Metrics score_metrics = compare(expected.scores, actual.scores);
                const Metrics softmax_metrics = compare(expected.softmax, actual.softmax);
                const Metrics av_metrics = compare(expected.output, actual.output);
                std::cout << ',' << score_metrics.max_abs << ',' << score_metrics.relative_l2 << ','
                          << softmax_metrics.max_abs << ',' << softmax_metrics.relative_l2 << ','
                          << av_metrics.max_abs << ',' << av_metrics.relative_l2;
            } else {
                std::cout << ",NA,NA,NA,NA,NA,NA";
            }
            std::cout << '\n';
        }
        std::cout << "PASS: historical-only OscarInt2G128 fused K/QK + V/AV path is reference-correct "
                     "at 128/512/2K/4K and benchmarked through 32K\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
