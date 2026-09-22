#include "exl3/linear_cuda.h"
#include "exl3/safetensors.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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
using namespace ninfer::exl3;

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

const char* env(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? "" : value;
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct DeviceBuffer {
    void* ptr = nullptr;
    ~DeviceBuffer() { if (ptr != nullptr) cudaFree(ptr); }
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

TensorPayload load(const IndexedSafetensors& collection, const std::string& name) {
    for (const auto& shard : collection.shards) {
        if (shard.header.find(name) != nullptr) return read_tensor(shard.path, shard.header, name);
    }
    throw std::runtime_error("missing tensor: " + name);
}

TensorPayload load_file_tensor(const std::filesystem::path& path, const std::string& name) {
    return read_tensor(path, inspect_file(path), name);
}

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (bits & 0x8000u) << 16u;
    const std::uint32_t exponent = (bits >> 10u) & 31u;
    const std::uint32_t fraction = bits & 1023u;
    std::uint32_t value = sign;
    if (exponent == 0) {
        if (fraction != 0) {
            std::uint32_t mantissa = fraction;
            int shift = 0;
            while ((mantissa & 1024u) == 0) { mantissa <<= 1u; ++shift; }
            value |= static_cast<std::uint32_t>(127 - 14 - shift) << 23u;
            value |= (mantissa & 1023u) << 13u;
        }
    } else if (exponent == 31) {
        value |= 0x7f800000u | (fraction << 13u);
    } else {
        value |= (exponent + 112u) << 23u | (fraction << 13u);
    }
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

std::uint16_t float_to_half(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    const std::uint32_t exponent = (bits >> 23u) & 255u;
    const std::uint32_t fraction = bits & 0x7fffffu;
    if (exponent == 255u) return static_cast<std::uint16_t>(sign | (fraction ? 0x7e00u : 0x7c00u));
    const int unbiased = static_cast<int>(exponent) - 127;
    if (unbiased > 15) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (unbiased >= -14) {
        std::uint32_t half_exp = static_cast<std::uint32_t>(unbiased + 15);
        std::uint32_t half_frac = fraction >> 13u;
        const std::uint32_t remainder = fraction & 0x1fffu;
        if (remainder > 0x1000u || (remainder == 0x1000u && (half_frac & 1u))) {
            ++half_frac;
            if (half_frac == 0x400u) { half_frac = 0; ++half_exp; }
        }
        return half_exp >= 31u ? static_cast<std::uint16_t>(sign | 0x7c00u) :
            static_cast<std::uint16_t>(sign | (half_exp << 10u) | half_frac);
    }
    if (unbiased < -25) return static_cast<std::uint16_t>(sign);
    const std::uint32_t mantissa = fraction | 0x00800000u;
    const int shift = -unbiased - 14;
    std::uint32_t half_frac = mantissa >> (shift + 13);
    const std::uint32_t remainder_mask = (1u << (shift + 13)) - 1u;
    const std::uint32_t remainder = mantissa & remainder_mask;
    const std::uint32_t halfway = 1u << (shift + 12);
    if (remainder > halfway || (remainder == halfway && (half_frac & 1u))) ++half_frac;
    return static_cast<std::uint16_t>(sign | half_frac);
}

std::vector<std::uint16_t> f16_values(const TensorPayload& tensor) {
    if (tensor.info.dtype == "F16") {
        const auto values = tensor.typed<std::uint16_t>("F16");
        return std::vector<std::uint16_t>(values.begin(), values.end());
    }
    require(tensor.info.dtype == "F32", tensor.info.name + " must be F16 or F32");
    const auto values = tensor.typed<float>("F32");
    std::vector<std::uint16_t> result;
    result.reserve(values.size());
    for (float value : values) result.push_back(float_to_half(value));
    return result;
}

struct Linear {
    std::vector<TensorPayload> host;
    std::vector<std::unique_ptr<DeviceBuffer>> device;
    Exl3CudaLinearWeights view{};
    Exl3CudaLinearMetadata metadata{};
};

Linear make_linear(const IndexedSafetensors& collection, const std::string& prefix) {
    Linear result;
    for (const char* suffix : {"trellis", "suh", "svh", "mul1"}) {
        result.host.push_back(load(collection, prefix + "." + suffix));
    }
    auto upload = [&](const TensorPayload& tensor, const char* label) {
        auto buffer = std::make_unique<DeviceBuffer>();
        check(cudaMalloc(&buffer->ptr, tensor.bytes().size_bytes()), label);
        check(cudaMemcpy(buffer->ptr, tensor.bytes().data(), tensor.bytes().size_bytes(), cudaMemcpyHostToDevice),
              "upload large-down tensor");
        void* pointer = buffer->ptr;
        result.device.push_back(std::move(buffer));
        return pointer;
    };
    result.view.trellis = static_cast<const std::uint16_t*>(upload(result.host[0], "allocate trellis"));
    result.view.suh = static_cast<const std::uint16_t*>(upload(result.host[1], "allocate suh"));
    result.view.svh = static_cast<const std::uint16_t*>(upload(result.host[2], "allocate svh"));
    result.view.mul1 = static_cast<const std::int32_t*>(upload(result.host[3], "allocate mul1"));
    require(result.host[0].info.shape.size() == 3, prefix + " trellis rank mismatch");
    result.metadata = {
        static_cast<int>(result.host[0].info.shape[0] * 16),
        static_cast<int>(result.host[0].info.shape[1] * 16),
        static_cast<int>(result.host[0].info.shape[2] / 16),
        false, true, false};
    require(result.metadata.in_features == 17408 && result.metadata.out_features == 5120,
            prefix + " is not the required large-down shape");
    return result;
}

std::vector<std::uint16_t> deterministic_input(int seed) {
    std::vector<std::uint16_t> values(17408);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const int phase = static_cast<int>((i + static_cast<std::size_t>(seed) * 17) % 23);
        const float magnitude = 0.015f * static_cast<float>(phase - 11);
        values[i] = float_to_half(magnitude + ((i % 97) == 0 ? (seed & 1 ? 1.5f : -1.25f) : 0.0f));
    }
    return values;
}

std::vector<std::uint16_t> run(Linear& linear, const std::vector<std::uint16_t>& input,
                               const std::string& topology) {
    require(input.size() == 17408, "large-down input length mismatch");
    DeviceBuffer input_device, output_device;
    check(cudaMalloc(&input_device.ptr, input.size() * sizeof(std::uint16_t)), "allocate oracle input");
    check(cudaMalloc(&output_device.ptr, 5120u * sizeof(std::uint16_t)), "allocate oracle output");
    check(cudaMemcpy(input_device.ptr, input.data(), input.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice),
          "upload oracle input");
    // The topology is intentionally selected by the library from this process
    // environment; "32" is the explicit inherited differential baseline.
    _putenv_s("NINFER_EXL3_LARGE_DOWN_TOPOLOGY", topology.c_str());
    Exl3CudaLinearWorkspace workspace(17408, 5120, 1);
    workspace.forward(linear.view, linear.metadata,
                      static_cast<const std::uint16_t*>(input_device.ptr),
                      static_cast<std::uint16_t*>(output_device.ptr), 1);
    check(cudaDeviceSynchronize(), "synchronize large-down oracle");
    std::vector<std::uint16_t> output(5120);
    check(cudaMemcpy(output.data(), output_device.ptr, output.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
          "download large-down oracle");
    return output;
}

struct Metrics { double max_abs = 0.0; double rel = 0.0; std::size_t index = 0; };

Metrics compare(const std::vector<std::uint16_t>& reference, const std::vector<std::uint16_t>& actual) {
    require(reference.size() == actual.size(), "large-down output size mismatch");
    double sum_sq = 0.0, ref_sq = 0.0;
    Metrics result;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double expected = half_to_float(reference[i]);
        const double observed = half_to_float(actual[i]);
        const double error = std::abs(observed - expected);
        if (error > result.max_abs) { result.max_abs = error; result.index = i; }
        const double delta = observed - expected;
        sum_sq += delta * delta;
        ref_sq += expected * expected;
    }
    result.rel = std::sqrt(sum_sq / std::max(ref_sq, 1e-30));
    return result;
}

} // namespace

int main() {
    try {
        const std::string target = env("NINFER_EXL3_TARGET_PATH");
        if (target.empty()) { std::cerr << "SKIP: set NINFER_EXL3_TARGET_PATH\n"; return 77; }
        const std::string e3a_path = env("NINFER_EXL3_E3A_ORACLE_PATH");
        const std::string e3b_path = env("NINFER_EXL3_E3B_ORACLE_PATH");
        if (e3a_path.empty() || e3b_path.empty()) {
            std::cerr << "SKIP: set NINFER_EXL3_E3A_ORACLE_PATH and NINFER_EXL3_E3B_ORACLE_PATH\n";
            return 77;
        }
        const auto target_collection = inspect_indexed_directory(std::filesystem::path(target));
        const auto e3a_header = inspect_file(e3a_path);
        const auto e3b_header = inspect_file(e3b_path);
        std::vector<std::pair<std::string, std::string>> cases = {
            {"full_k6_real", "model.language_model.layers.19.mlp.down_proj"},
            {"gdn_k6_real", "model.language_model.layers.5.mlp.down_proj"},
            {"gdn_k7_deterministic", "model.language_model.layers.23.mlp.down_proj"}};
        // The canonical schedule has 16 full-attention layers. Locate a real
        // K7 down tensor instead of assuming that layer 63 has that metadata.
        const int full_layers[] = {3, 7, 11, 15, 19, 23, 27, 31,
                                   35, 39, 43, 47, 51, 55, 59, 63};
        for (int layer : full_layers) {
            const std::string prefix = "model.language_model.layers." +
                std::to_string(layer) + ".mlp.down_proj";
            const TensorInfo* trellis = nullptr;
            for (const auto& shard : target_collection.shards) {
                trellis = shard.header.find(prefix + ".trellis");
                if (trellis != nullptr) break;
            }
            require(trellis != nullptr, prefix + " trellis is absent");
            require(trellis->shape.size() == 3, prefix + " trellis rank mismatch");
            if (trellis->shape[2] / 16 == 7) {
                cases.emplace_back("full_k7_deterministic", prefix);
                break;
            }
        }
        require(cases.size() == 4, "canonical target has no real full-attention K7 down tensor");
        std::vector<std::vector<std::uint16_t>> inputs;
        inputs.push_back(f16_values(load_file_tensor(e3a_path, "m1_activated_mlp")));
        inputs.push_back(f16_values(load_file_tensor(e3b_path, "decode0_down_projection_input")));
        inputs.push_back(deterministic_input(3));
        inputs.push_back(deterministic_input(7));
        require(inputs[0].size() == 17408 && inputs[1].size() == 17408, "real fixture activation shape mismatch");

        bool failed = false;
        for (std::size_t i = 0; i < cases.size(); ++i) {
            auto linear = make_linear(target_collection, cases[i].second);
            require(linear.metadata.K == 6 || linear.metadata.K == 7, cases[i].second + " is not K6/K7");
            const auto baseline = run(linear, inputs[i], "32");
            for (const std::string topology : {std::string("16"), std::string("8")}) {
                const auto candidate = run(linear, inputs[i], topology);
                const auto metrics = compare(baseline, candidate);
                const std::size_t tile = metrics.index / 16;
                const std::size_t lane = metrics.index % 16;
                std::cout << "ORACLE case=" << cases[i].first << " topology=" << topology
                          << " K=" << linear.metadata.K
                          << " max_abs=" << std::setprecision(10) << metrics.max_abs
                          << " rel_l2=" << metrics.rel << " first_index=" << metrics.index
                          << " first_tile=" << tile << " first_lane=" << lane << '\n';
                if (metrics.max_abs > 0.0015 || metrics.rel > 0.0007) failed = true;
            }
        }
        _putenv_s("NINFER_EXL3_LARGE_DOWN_TOPOLOGY", "");
        if (failed) throw std::runtime_error("large-down topology differential oracle failed");
        std::cout << "E4B3D large-down differential oracle: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "E4B3D large-down differential oracle: FAIL: " << error.what() << '\n';
        return 1;
    }
}
