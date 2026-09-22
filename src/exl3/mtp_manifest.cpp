#include "exl3/mtp_manifest.h"

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace ninfer::exl3 {
namespace {

using LocatedTensor = std::pair<std::size_t, const TensorInfo*>;

const std::array<Exl3MtpTensorSpec, kQwen38NativeMtpTensorCount> kManifest = {{
    // fc: 10240 -> 5120
    {"mtp.fc.trellis", "I16", {640, 320, 64}, 3, 26'214'400},
    {"mtp.fc.suh", "F16", {10'240, 0, 0}, 1, 20'480},
    {"mtp.fc.svh", "F16", {5'120, 0, 0}, 1, 10'240},
    {"mtp.fc.mul1", "I32", {0, 0, 0}, 0, 4},

    // q: 5120 -> 12288
    {"mtp.layers.0.self_attn.q_proj.trellis", "I16", {320, 768, 64}, 3,
     31'457'280},
    {"mtp.layers.0.self_attn.q_proj.suh", "F16", {5'120, 0, 0}, 1, 10'240},
    {"mtp.layers.0.self_attn.q_proj.svh", "F16", {12'288, 0, 0}, 1, 24'576},
    {"mtp.layers.0.self_attn.q_proj.mul1", "I32", {0, 0, 0}, 0, 4},

    // k: 5120 -> 1024
    {"mtp.layers.0.self_attn.k_proj.trellis", "I16", {320, 64, 64}, 3,
     2'621'440},
    {"mtp.layers.0.self_attn.k_proj.suh", "F16", {5'120, 0, 0}, 1, 10'240},
    {"mtp.layers.0.self_attn.k_proj.svh", "F16", {1'024, 0, 0}, 1, 2'048},
    {"mtp.layers.0.self_attn.k_proj.mul1", "I32", {0, 0, 0}, 0, 4},

    // v: 5120 -> 1024
    {"mtp.layers.0.self_attn.v_proj.trellis", "I16", {320, 64, 64}, 3,
     2'621'440},
    {"mtp.layers.0.self_attn.v_proj.suh", "F16", {5'120, 0, 0}, 1, 10'240},
    {"mtp.layers.0.self_attn.v_proj.svh", "F16", {1'024, 0, 0}, 1, 2'048},
    {"mtp.layers.0.self_attn.v_proj.mul1", "I32", {0, 0, 0}, 0, 4},

    // o: 6144 -> 5120
    {"mtp.layers.0.self_attn.o_proj.trellis", "I16", {384, 320, 64}, 3,
     15'728'640},
    {"mtp.layers.0.self_attn.o_proj.suh", "F16", {6'144, 0, 0}, 1, 12'288},
    {"mtp.layers.0.self_attn.o_proj.svh", "F16", {5'120, 0, 0}, 1, 10'240},
    {"mtp.layers.0.self_attn.o_proj.mul1", "I32", {0, 0, 0}, 0, 4},

    // gate: 5120 -> 17408
    {"mtp.layers.0.mlp.gate_proj.trellis", "I16", {320, 1088, 64}, 3,
     44'564'480},
    {"mtp.layers.0.mlp.gate_proj.suh", "F16", {5'120, 0, 0}, 1, 10'240},
    {"mtp.layers.0.mlp.gate_proj.svh", "F16", {17'408, 0, 0}, 1, 34'816},
    {"mtp.layers.0.mlp.gate_proj.mul1", "I32", {0, 0, 0}, 0, 4},

    // up: 5120 -> 17408
    {"mtp.layers.0.mlp.up_proj.trellis", "I16", {320, 1088, 64}, 3,
     44'564'480},
    {"mtp.layers.0.mlp.up_proj.suh", "F16", {5'120, 0, 0}, 1, 10'240},
    {"mtp.layers.0.mlp.up_proj.svh", "F16", {17'408, 0, 0}, 1, 34'816},
    {"mtp.layers.0.mlp.up_proj.mul1", "I32", {0, 0, 0}, 0, 4},

    // down: 17408 -> 5120
    {"mtp.layers.0.mlp.down_proj.trellis", "I16", {1088, 320, 64}, 3,
     44'564'480},
    {"mtp.layers.0.mlp.down_proj.suh", "F16", {17'408, 0, 0}, 1, 34'816},
    {"mtp.layers.0.mlp.down_proj.svh", "F16", {5'120, 0, 0}, 1, 10'240},
    {"mtp.layers.0.mlp.down_proj.mul1", "I32", {0, 0, 0}, 0, 4},

    // Layer and input/output norms remain BF16 in the native source set.
    {"mtp.layers.0.input_layernorm.weight", "BF16", {5'120, 0, 0}, 1,
     10'240},
    {"mtp.layers.0.post_attention_layernorm.weight", "BF16", {5'120, 0, 0},
     1, 10'240},
    {"mtp.layers.0.self_attn.k_norm.weight", "BF16", {256, 0, 0}, 1, 512},
    {"mtp.layers.0.self_attn.q_norm.weight", "BF16", {256, 0, 0}, 1, 512},
    {"mtp.norm.weight", "BF16", {5'120, 0, 0}, 1, 10'240},
    {"mtp.pre_fc_norm_embedding.weight", "BF16", {5'120, 0, 0}, 1,
     10'240},
    {"mtp.pre_fc_norm_hidden.weight", "BF16", {5'120, 0, 0}, 1, 10'240},
}};

[[noreturn]] void reject(std::string message) {
    throw SafetensorsError("Qwen3.8 native MTP manifest: " + std::move(message));
}

bool is_mtp_name(std::string_view name) noexcept {
    return name.size() >= 4 && name.substr(0, 4) == "mtp.";
}

const Exl3MtpTensorSpec* find_spec(std::string_view name) noexcept {
    for (const auto& spec : kManifest) {
        if (spec.name == name) return &spec;
    }
    return nullptr;
}

bool shape_matches(const Exl3MtpTensorSpec& spec,
                   const std::vector<std::uint64_t>& actual) noexcept {
    return actual.size() == spec.rank &&
           std::equal(actual.begin(), actual.end(), spec.shape.begin());
}

std::vector<LocatedTensor> validate_and_locate(const IndexedSafetensors& collection) {
    if (collection.shards.empty()) reject("indexed collection has no shards");
    if (collection.indexed_tensor_count != collection.header_tensor_count) {
        reject("indexed/header tensor counts differ");
    }

    std::unordered_map<std::string, LocatedTensor> located;
    located.reserve(kManifest.size());
    for (std::size_t shard_index = 0; shard_index < collection.shards.size(); ++shard_index) {
        const auto& shard = collection.shards[shard_index];
        for (const auto& tensor : shard.header.tensors) {
            if (!is_mtp_name(tensor.name)) continue;
            if (!located.emplace(tensor.name, LocatedTensor{shard_index, &tensor}).second) {
                reject("duplicate MTP tensor: " + tensor.name);
            }

            const auto* spec = find_spec(tensor.name);
            if (spec == nullptr) reject("unexpected MTP tensor: " + tensor.name);
            if (tensor.data_end < tensor.data_begin) {
                reject("MTP tensor has reversed data range: " + tensor.name);
            }
            if (tensor.dtype != spec->dtype) {
                reject("MTP tensor " + tensor.name + " has dtype " + tensor.dtype +
                       ", expected " + std::string(spec->dtype));
            }
            if (!shape_matches(*spec, tensor.shape)) {
                reject("MTP tensor " + tensor.name + " has the wrong shape");
            }
            if (tensor.bytes() != spec->bytes) {
                reject("MTP tensor " + tensor.name + " has " +
                       std::to_string(tensor.bytes()) + " bytes, expected " +
                       std::to_string(spec->bytes));
            }
        }
    }

    if (located.size() != kManifest.size()) {
        for (const auto& spec : kManifest) {
            if (!located.contains(std::string(spec.name))) {
                reject("missing MTP tensor: " + std::string(spec.name));
            }
        }
        reject("MTP tensor count does not match the closed manifest");
    }

    std::vector<LocatedTensor> ordered;
    ordered.reserve(kManifest.size());
    std::uint64_t total_bytes = 0;
    for (const auto& spec : kManifest) {
        const auto it = located.find(std::string(spec.name));
        if (it == located.end()) reject("missing MTP tensor: " + std::string(spec.name));
        if (spec.bytes > std::numeric_limits<std::uint64_t>::max() - total_bytes) {
            reject("MTP manifest byte total overflows u64");
        }
        total_bytes += spec.bytes;
        ordered.push_back(it->second);
    }
    if (total_bytes != kQwen38NativeMtpTotalBytes) {
        reject("built-in MTP manifest byte total is inconsistent");
    }
    return ordered;
}

} // namespace

const Exl3MtpTensorBinding* Exl3MtpBinding::find(std::string_view name) const noexcept {
    const auto it = std::find_if(tensors.begin(), tensors.end(),
                                 [name](const auto& tensor) { return tensor.name == name; });
    return it == tensors.end() ? nullptr : &*it;
}

std::span<const Exl3MtpTensorSpec> qwen3_8_27b_native_mtp_manifest() noexcept {
    return std::span<const Exl3MtpTensorSpec>(kManifest.data(), kManifest.size());
}

Exl3MtpManifestSummary validate_qwen3_8_27b_native_mtp(
    const IndexedSafetensors& collection) {
    (void)validate_and_locate(collection);
    return {kManifest.size(), kQwen38NativeMtpTotalBytes};
}

Exl3MtpBinding bind_qwen3_8_27b_native_mtp(const IndexedSafetensors& collection) {
    const auto ordered = validate_and_locate(collection);
    Exl3MtpBinding result;
    result.tensors.reserve(ordered.size());
    result.total_bytes = kQwen38NativeMtpTotalBytes;
    for (std::size_t i = 0; i < ordered.size(); ++i) {
        const auto [shard_index, info] = ordered[i];
        result.tensors.push_back(
            {kManifest[i].name, collection.shards[shard_index].path, *info});
    }
    return result;
}

} // namespace ninfer::exl3
