#include "exl3/linear_reference.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;
using ninfer::exl3::Exl3LinearMetadata;
using ninfer::exl3::Exl3LinearTensors;
using ninfer::exl3::Exl3ReferenceOutput;
using ninfer::exl3::SafetensorsError;
using ninfer::exl3::SafetensorsHeader;
using ninfer::exl3::TensorPayload;

constexpr const char* kPrefix = "model.language_model.layers.5.mlp.gate_proj.";
constexpr const char* kTrellisName = "model.language_model.layers.5.mlp.gate_proj.trellis";
constexpr const char* kSuhName = "model.language_model.layers.5.mlp.gate_proj.suh";
constexpr const char* kSvhName = "model.language_model.layers.5.mlp.gate_proj.svh";
constexpr const char* kMul1Name = "model.language_model.layers.5.mlp.gate_proj.mul1";
// E1's retained CUDA repeatability audit reached 0.00146484375 max-abs and
// 0.0006054458726 relative-L2 for this same tensor. The native reference is
// required to stay inside that observed envelope with a small representation
// margin, rather than using a broad model-level tolerance.
constexpr double kMaxAbsolutePass = 0.0015;
constexpr double kRelativeL2Pass = 7.0e-4;

std::string env_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
}

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

float half_to_float(std::uint16_t value) {
    const std::uint32_t sign = (value & 0x8000u) << 16u;
    const std::uint32_t exponent = (value >> 10u) & 0x1fu;
    const std::uint32_t fraction = value & 0x03ffu;
    std::uint32_t bits = sign;
    if (exponent == 0) {
        if (fraction != 0) {
            std::uint32_t normalized = fraction;
            std::uint32_t exp = 0;
            while ((normalized & 0x0400u) == 0) {
                normalized <<= 1u;
                ++exp;
            }
            bits |= (127u - 14u - exp) << 23u;
            bits |= (normalized & 0x03ffu) << 13u;
        }
    } else if (exponent == 0x1fu) {
        bits |= 0x7f800000u | (fraction << 13u);
    } else {
        bits |= (exponent + 112u) << 23u | (fraction << 13u);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

struct Metrics {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rms = 0.0;
    double relative_l2 = 0.0;
    double p99_abs = 0.0;
    double p999_abs = 0.0;
    std::size_t over_1e_3 = 0;
    std::size_t over_1e_2 = 0;
    std::size_t worst_index = 0;
    float actual_at_worst = 0.0f;
    float expected_at_worst = 0.0f;
};

Metrics compare(std::span<const std::uint16_t> actual,
                std::span<const std::uint16_t> expected) {
    require(actual.size() == expected.size(), "output element count mismatch");
    std::vector<double> errors;
    errors.reserve(actual.size());
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double ref_sq = 0.0;
    Metrics result;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float a = half_to_float(actual[i]);
        const float e = half_to_float(expected[i]);
        const double error = std::abs(static_cast<double>(a) - static_cast<double>(e));
        errors.push_back(error);
        sum_abs += error;
        sum_sq += error * error;
        ref_sq += static_cast<double>(e) * static_cast<double>(e);
        if (error > result.max_abs) {
            result.max_abs = error;
            result.worst_index = i;
            result.actual_at_worst = a;
            result.expected_at_worst = e;
        }
        if (error > 1.0e-3) { ++result.over_1e_3; }
        if (error > 1.0e-2) { ++result.over_1e_2; }
    }
    std::sort(errors.begin(), errors.end());
    const auto percentile = [&](double p) {
        if (errors.empty()) { return 0.0; }
        const std::size_t index = std::min(errors.size() - 1,
                                           static_cast<std::size_t>(p * (errors.size() - 1)));
        return errors[index];
    };
    result.mean_abs = sum_abs / static_cast<double>(actual.size());
    result.rms = std::sqrt(sum_sq / static_cast<double>(actual.size()));
    result.relative_l2 = ref_sq == 0.0 ? std::sqrt(sum_sq) : std::sqrt(sum_sq / ref_sq);
    result.p99_abs = percentile(0.99);
    result.p999_abs = percentile(0.999);
    return result;
}

void print_metrics(const char* label, const Metrics& metrics) {
    std::cout << label << ": max_abs=" << std::setprecision(10) << metrics.max_abs
              << " mean_abs=" << metrics.mean_abs
              << " rms=" << metrics.rms
              << " relative_l2=" << metrics.relative_l2
              << " p99_abs=" << metrics.p99_abs
              << " p999_abs=" << metrics.p999_abs
              << " over_1e-3=" << metrics.over_1e_3
              << " over_1e-2=" << metrics.over_1e_2
              << " worst_index=" << metrics.worst_index
              << " actual=" << metrics.actual_at_worst
              << " expected=" << metrics.expected_at_worst << '\n';
}

Json read_json(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "cannot open JSON: " + path.string());
    try {
        return Json::parse(std::string(std::istreambuf_iterator<char>(stream), {}));
    } catch (const Json::exception& exception) {
        throw std::runtime_error("invalid JSON " + path.string() + ": " + exception.what());
    }
}

TensorPayload load_indexed_tensor(const ninfer::exl3::IndexedSafetensors& collection,
                                  std::string_view name) {
    for (const auto& shard : collection.shards) {
        if (shard.header.find(name) != nullptr) {
            return ninfer::exl3::read_tensor(shard.path, shard.header, name);
        }
    }
    throw std::runtime_error("indexed tensor is absent: " + std::string(name));
}

void verify_model_config(const std::filesystem::path& model_dir) {
    const Json config = read_json(model_dir / "config.json");
    require(config.at("architectures").at(0).get<std::string>() == "Qwen3_5ForConditionalGeneration",
            "unexpected model architecture");
    const auto& text = config.at("text_config");
    require(text.at("hidden_size").get<int>() == 5120, "unexpected text hidden size");
    require(text.at("num_hidden_layers").get<int>() == 64, "unexpected text layer count");
    require(text.at("intermediate_size").get<int>() == 17408, "unexpected text intermediate size");
    const auto& quant = config.at("quantization_config");
    require(quant.at("quant_method").get<std::string>() == "exl3", "model is not EXL3");
    require(quant.at("version").get<std::string>() == "1.4.2", "unexpected EXL3 version");
    require(quant.at("bits").get<double>() == 6.0, "model bitrate is not 6.0");
    require(quant.at("head_bits").get<int>() == 6, "model H head is not 6 bpw");
    require(quant.at("vision_bits").get<int>() == 6, "model V head is not 6 bpw");
    require(quant.at("codebook").get<std::string>() == "mul1", "unexpected model codebook");
    require(quant.at("out_scales").get<std::string>() == "always", "unexpected output-scale policy");
    const auto& layer_types = text.at("layer_types");
    const auto full_count = std::count(layer_types.begin(), layer_types.end(), "full_attention");
    const auto linear_count = std::count(layer_types.begin(), layer_types.end(), "linear_attention");
    require(full_count == 16 && linear_count == 48, "unexpected hybrid layer schedule");
}

Exl3LinearTensors load_module(const std::filesystem::path& model_dir,
                              std::vector<TensorPayload>& keep_alive) {
    const auto collection = ninfer::exl3::inspect_indexed_directory(model_dir);
    const auto load = [&](const char* name) {
        keep_alive.push_back(load_indexed_tensor(collection, name));
        return &keep_alive.back();
    };
    Exl3LinearTensors tensors;
    tensors.trellis = load(kTrellisName);
    tensors.suh = load(kSuhName);
    tensors.svh = load(kSvhName);
    tensors.mul1 = load(kMul1Name);
    return tensors;
}

void expect_failure(const char* label, const std::function<void()>& action) {
    try {
        action();
    } catch (const SafetensorsError&) {
        std::cout << "negative PASS: " << label << '\n';
        return;
    }
    throw std::runtime_error(std::string("negative test unexpectedly succeeded: ") + label);
}

void run_negative_tests(const Exl3LinearTensors& original,
                        const Exl3LinearMetadata& metadata,
                        const TensorPayload& input) {
    auto check = [&](const char* label, Exl3LinearTensors candidate,
                     Exl3LinearMetadata candidate_metadata,
                     const TensorPayload* candidate_input) {
        expect_failure(label, [&] {
            if (std::string(label) == "payload outside shard bounds") {
                SafetensorsHeader header = ninfer::exl3::inspect_file(
                    env_or_empty("NINFER_EXL3_ORACLE_PATH"));
                header.tensors.front().data_end = header.payload_bytes + 1;
                (void)ninfer::exl3::read_tensor(env_or_empty("NINFER_EXL3_ORACLE_PATH"),
                                                header, header.tensors.front().name);
            } else {
                (void)ninfer::exl3::exl3_linear_reference(*candidate_input, candidate,
                                                          candidate_metadata);
            }
        });
    };

    {
        auto value = *original.trellis;
        value.info.dtype = "F16";
        check("wrong trellis dtype", {&value, original.suh, original.svh, original.mul1}, metadata, &input);
    }
    {
        auto value = *original.trellis;
        value.info.shape[2] = 64;
        check("wrong trellis shape", {&value, original.suh, original.svh, original.mul1}, metadata, &input);
    }
    check("missing suh", {original.trellis, nullptr, original.svh, original.mul1}, metadata, &input);
    check("missing svh", {original.trellis, original.suh, nullptr, original.mul1}, metadata, &input);
    check("missing mul1 when required", {original.trellis, original.suh, original.svh, nullptr}, metadata, &input);
    {
        auto bad = metadata;
        bad.K = 6;
        check("unsupported module K", original, bad, &input);
    }
    {
        auto bad = metadata;
        bad.codebook = "mcg";
        check("unsupported codebook", original, bad, &input);
    }
    {
        auto bad = metadata;
        bad.has_bias = true;
        check("unsupported bias", original, bad, &input);
    }
    {
        auto bad = input;
        bad.info.shape.back() = 4096;
        check("incompatible input dimension", original, metadata, &bad);
    }
    check("payload outside shard bounds", original, metadata, &input);
    std::cout << "negative tests: PASS\n";
}

std::optional<Metrics> compare_optional(const std::filesystem::path& path,
                                        const Exl3ReferenceOutput& actual) {
    if (path.empty() || !std::filesystem::exists(path)) { return std::nullopt; }
    const auto header = ninfer::exl3::inspect_file(path);
    const auto expected = ninfer::exl3::read_tensor(path, header, "linear_output");
    const auto bits = expected.typed<std::uint16_t>("F16");
    return compare(actual.fp16_bits, bits);
}

} // namespace

int main() {
    try {
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
        verify_model_config(model_dir);

        std::vector<TensorPayload> keep_alive;
        keep_alive.reserve(4);
        const auto tensors = load_module(model_dir, keep_alive);
        const auto mul1_value = tensors.mul1->typed<std::int32_t>("I32").front();
        std::cout << "module mul1 scalar=" << mul1_value << '\n';
        const auto oracle_header = ninfer::exl3::inspect_file(oracle_path);
        const auto input = ninfer::exl3::read_tensor(oracle_path, oracle_header, "linear_input");
        const auto expected = ninfer::exl3::read_tensor(oracle_path, oracle_header, "linear_output");
        require(input.info.shape == std::vector<std::uint64_t>{1, 14, 5120},
                "oracle linear_input shape mismatch");
        require(input.info.dtype == "F16", "oracle linear_input dtype mismatch");
        require(expected.info.shape == std::vector<std::uint64_t>{1, 14, 17408},
                "oracle linear_output shape mismatch");
        require(expected.info.dtype == "F16", "oracle linear_output dtype mismatch");

        const Exl3LinearMetadata metadata{5120, 17408, 5, false, true, false, "mul1"};
        ninfer::exl3::validate_linear_tensors(tensors, metadata);
        const auto actual = ninfer::exl3::exl3_linear_reference(input, tensors, metadata);
        require(actual.shape == std::vector<std::uint64_t>{1, 14, 17408},
                "native reference output shape mismatch");
        const auto expected_bits = expected.typed<std::uint16_t>("F16");
        const auto metrics = compare(actual.fp16_bits, expected_bits);
        print_metrics("canonical linear_output", metrics);

        const auto run2 = compare_optional(
            env_or_empty("NINFER_EXL3_ORACLE_RUN2_PATH"), actual);
        const auto run3 = compare_optional(
            env_or_empty("NINFER_EXL3_ORACLE_RUN3_PATH"), actual);
        if (run2) { print_metrics("run2 linear_output", *run2); }
        if (run3) { print_metrics("run3 linear_output", *run3); }

        run_negative_tests(tensors, metadata, input);
        require(metrics.max_abs <= kMaxAbsolutePass,
                "canonical EXL3 output exceeds max-absolute tolerance");
        require(metrics.relative_l2 <= kRelativeL2Pass,
                "canonical EXL3 output exceeds relative-L2 tolerance");
        std::cout << "E2 single-tensor correctness: PASS\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "E2 single-tensor correctness: FAIL: " << exception.what() << '\n';
        return 1;
    }
}
