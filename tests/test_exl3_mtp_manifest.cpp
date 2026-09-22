#include "exl3/mtp_manifest.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::exl3::Exl3MtpTensorSpec;
using ninfer::exl3::IndexedSafetensors;
using ninfer::exl3::SafetensorsError;
using ninfer::exl3::TensorInfo;

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

template <typename Function>
void expect_rejected(Function&& function, std::string_view label) {
    try {
        function();
    } catch (const SafetensorsError&) {
        return;
    }
    throw std::runtime_error(std::string(label) + " was accepted");
}

TensorInfo descriptor(const Exl3MtpTensorSpec& spec, std::uint64_t offset) {
    TensorInfo info;
    info.name = std::string(spec.name);
    info.dtype = std::string(spec.dtype);
    info.shape.assign(spec.dimensions().begin(), spec.dimensions().end());
    info.data_begin = offset;
    info.data_end = offset + spec.bytes;
    return info;
}

IndexedSafetensors valid_collection() {
    IndexedSafetensors collection;
    IndexedSafetensors::Shard shard;
    shard.path = std::filesystem::path("missing-native-mtp-shard.safetensors");
    std::uint64_t offset = 0;
    for (const auto& spec : ninfer::exl3::qwen3_8_27b_native_mtp_manifest()) {
        shard.header.tensors.push_back(descriptor(spec, offset));
        offset += spec.bytes;
    }
    // The native model also contains non-MTP tensors.  They are intentionally
    // ignored by this closed MTP sub-manifest.
    shard.header.tensors.push_back(
        TensorInfo{"model.layers.0.self_attn.q_proj.weight", "I16", {1}, offset,
                   offset + 2});
    collection.shards.push_back(std::move(shard));
    collection.indexed_tensor_count = collection.shards.front().header.tensors.size();
    collection.header_tensor_count = collection.indexed_tensor_count;
    return collection;
}

TensorInfo& require_tensor(IndexedSafetensors& collection, std::string_view name) {
    for (auto& tensor : collection.shards.front().header.tensors) {
        if (tensor.name == name) return tensor;
    }
    throw std::runtime_error("fixture tensor is absent");
}

float bf16_to_float(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16u);
}

float half_to_float(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16u;
    std::uint32_t exponent = (value >> 10u) & 0x1fu;
    std::uint32_t fraction = value & 0x03ffu;
    std::uint32_t bits = sign;
    if (exponent == 0) {
        if (fraction) {
            unsigned shift = 0;
            while ((fraction & 0x0400u) == 0) { fraction <<= 1u; ++shift; }
            bits |= (127u - 14u - shift) << 23u;
            bits |= (fraction & 0x03ffu) << 13u;
        }
    } else if (exponent == 0x1fu) {
        bits |= 0x7f800000u | (fraction << 13u);
    } else {
        bits |= (exponent + 112u) << 23u | (fraction << 13u);
    }
    return std::bit_cast<float>(bits);
}

std::uint16_t float_to_half(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    const std::uint32_t absolute = bits & 0x7fffffffu;
    if (absolute >= 0x7f800000u)
        return static_cast<std::uint16_t>(sign | (absolute > 0x7f800000u ? 0x7e00u : 0x7c00u));
    int exponent = static_cast<int>((absolute >> 23u) & 0xffu) - 127 + 15;
    std::uint32_t mantissa = absolute & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<std::uint16_t>(sign);
        mantissa |= 0x800000u;
        const unsigned shift = static_cast<unsigned>(14 - exponent);
        std::uint32_t rounded = mantissa >> shift;
        const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    std::uint32_t rounded = mantissa >> 13u;
    const std::uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        if (++rounded == 0x400u) {
            rounded = 0;
            if (++exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<unsigned>(exponent) << 10u) | rounded);
}

void require_live_norms_exact_in_f16(const ninfer::exl3::Exl3MtpBinding& binding) {
    std::size_t tensors = 0, values = 0;
    for (const auto& tensor : binding.tensors) {
        if (tensor.info.dtype != "BF16") continue;
        const auto payload = ninfer::exl3::read_tensor(tensor.shard_path, tensor.info.name);
        for (const auto value : payload.typed<std::uint16_t>("BF16")) {
            const float represented = bf16_to_float(value);
            require(std::isfinite(represented), "live MTP norm contains non-finite BF16");
            require(half_to_float(float_to_half(represented)) == represented,
                    "live MTP BF16 norm is not exactly representable in F16");
            ++values;
        }
        ++tensors;
    }
    require(tensors == 7 && values == 26'112, "live MTP norm inventory mismatch");
    std::cout << "live_mtp_f16_exact_norm_tensors=" << tensors
              << " live_mtp_f16_exact_norm_values=" << values << '\n';
}

void test_exact_manifest_and_header_only_binding() {
    const auto manifest = ninfer::exl3::qwen3_8_27b_native_mtp_manifest();
    require(manifest.size() == 39, "native MTP manifest count drifted");
    require(ninfer::exl3::kQwen38NativeMtpTensorCount == 39,
            "native MTP count constant drifted");
    require(ninfer::exl3::kQwen38NativeMtpTotalBytes == 212'636'704,
            "native MTP byte total drifted");

    const auto find = [&](std::string_view name) -> const Exl3MtpTensorSpec* {
        for (const auto& spec : manifest) {
            if (spec.name == name) return &spec;
        }
        return nullptr;
    };
    const auto* fc = find("mtp.fc.trellis");
    require(fc != nullptr && fc->dtype == "I16" &&
                std::vector<std::uint64_t>(fc->dimensions().begin(), fc->dimensions().end()) ==
                    std::vector<std::uint64_t>{640, 320, 64} && fc->bytes == 26'214'400,
            "fc K4 descriptor drifted");
    const auto* q = find("mtp.layers.0.self_attn.q_proj.trellis");
    require(q != nullptr && q->dtype == "I16" &&
                std::vector<std::uint64_t>(q->dimensions().begin(), q->dimensions().end()) ==
                    std::vector<std::uint64_t>{320, 768, 64} && q->bytes == 31'457'280,
            "q K4 descriptor drifted");
    const auto* norm = find("mtp.layers.0.input_layernorm.weight");
    require(norm != nullptr && norm->dtype == "BF16" && norm->rank == 1 &&
                norm->shape[0] == 5120 && norm->bytes == 10'240,
            "BF16 norm descriptor drifted");

    const auto collection = valid_collection();
    const auto summary = ninfer::exl3::validate_qwen3_8_27b_native_mtp(collection);
    require(summary.tensor_count == 39 && summary.total_bytes == 212'636'704,
            "valid native MTP summary mismatch");
    const auto binding = ninfer::exl3::bind_qwen3_8_27b_native_mtp(collection);
    require(binding.tensors.size() == 39 && binding.total_bytes == 212'636'704,
            "valid native MTP binding mismatch");
    require(binding.tensors.front().name == "mtp.fc.trellis" &&
                binding.tensors.back().name == "mtp.pre_fc_norm_hidden.weight",
            "native MTP binding order mismatch");
    const auto* q_binding = binding.find("mtp.layers.0.self_attn.q_proj.trellis");
    require(q_binding != nullptr && q_binding->shard_path.filename() ==
                                        "missing-native-mtp-shard.safetensors" &&
                q_binding->info.bytes() == 31'457'280,
            "native MTP binding lookup mismatch");
}

void test_fail_closed_cases() {
    auto missing = valid_collection();
    auto& missing_tensor = missing.shards.front().header.tensors;
    missing_tensor.erase(
        std::find_if(missing_tensor.begin(), missing_tensor.end(), [](const auto& tensor) {
            return tensor.name == "mtp.fc.mul1";
        }));
    --missing.indexed_tensor_count;
    --missing.header_tensor_count;
    expect_rejected([&] { (void)ninfer::exl3::bind_qwen3_8_27b_native_mtp(missing); },
                    "missing MTP tensor");

    auto extra = valid_collection();
    extra.shards.front().header.tensors.push_back(
        TensorInfo{"mtp.unexpected", "BF16", {1}, 212'636'704, 212'636'706});
    ++extra.indexed_tensor_count;
    ++extra.header_tensor_count;
    expect_rejected([&] { (void)ninfer::exl3::bind_qwen3_8_27b_native_mtp(extra); },
                    "extra MTP tensor");

    auto wrong_dtype = valid_collection();
    require_tensor(wrong_dtype, "mtp.fc.trellis").dtype = "F16";
    expect_rejected([&] { (void)ninfer::exl3::bind_qwen3_8_27b_native_mtp(wrong_dtype); },
                    "wrong MTP dtype");

    auto wrong_shape = valid_collection();
    require_tensor(wrong_shape, "mtp.fc.trellis").shape[0]++;
    expect_rejected([&] { (void)ninfer::exl3::bind_qwen3_8_27b_native_mtp(wrong_shape); },
                    "wrong MTP shape");

    auto wrong_bytes = valid_collection();
    ++require_tensor(wrong_bytes, "mtp.fc.trellis").data_end;
    expect_rejected([&] { (void)ninfer::exl3::bind_qwen3_8_27b_native_mtp(wrong_bytes); },
                    "wrong MTP byte count");

    auto reversed_range = valid_collection();
    auto& reversed = require_tensor(reversed_range, "mtp.fc.suh");
    reversed.data_begin = reversed.data_end + 1;
    expect_rejected([&] { (void)ninfer::exl3::bind_qwen3_8_27b_native_mtp(reversed_range); },
                    "reversed MTP range");

    auto duplicate = valid_collection();
    duplicate.shards.push_back(duplicate.shards.front());
    duplicate.indexed_tensor_count *= 2;
    duplicate.header_tensor_count *= 2;
    expect_rejected([&] { (void)ninfer::exl3::bind_qwen3_8_27b_native_mtp(duplicate); },
                    "duplicate MTP tensor");

    auto bad_counts = valid_collection();
    --bad_counts.indexed_tensor_count;
    expect_rejected([&] { (void)ninfer::exl3::bind_qwen3_8_27b_native_mtp(bad_counts); },
                    "inconsistent indexed/header counts");
}

} // namespace

int main() {
    try {
        test_exact_manifest_and_header_only_binding();
        test_fail_closed_cases();
        if (const char* model = std::getenv("NINFER_EXL3_TARGET_PATH");
            model != nullptr && *model != '\0') {
            const auto collection = ninfer::exl3::inspect_indexed_directory(model);
            const auto binding = ninfer::exl3::bind_qwen3_8_27b_native_mtp(collection);
            require(binding.tensors.size() == 39 &&
                        binding.total_bytes == 212'636'704,
                    "live native MTP binding mismatch");
            require_live_norms_exact_in_f16(binding);
            std::cout << "live_mtp_tensors=" << binding.tensors.size()
                      << " live_mtp_bytes=" << binding.total_bytes << '\n';
        }
        std::cout << "exl3_mtp_manifest_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "exl3_mtp_manifest_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
