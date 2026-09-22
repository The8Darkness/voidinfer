#pragma once

#include "exl3/safetensors.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::exl3 {

// Native Qwen3.8-27B EXL3 MTP is deliberately a closed manifest.  The
// descriptor is for the source safetensors set, not for a materialized
// .ninfer object, and therefore retains the native tensor names and dtypes.
inline constexpr std::size_t kQwen38NativeMtpTensorCount = 39;
inline constexpr std::uint64_t kQwen38NativeMtpTotalBytes = 212'636'704;

struct Exl3MtpTensorSpec {
    std::string_view name;
    std::string_view dtype;
    std::array<std::uint64_t, 3> shape{};
    std::size_t rank = 0;
    std::uint64_t bytes = 0;

    [[nodiscard]] std::span<const std::uint64_t> dimensions() const noexcept {
        return std::span<const std::uint64_t>(shape.data(), rank);
    }
};

// One validated metadata binding.  It intentionally contains only a copied
// descriptor and the shard path; obtaining bytes remains an explicit later
// read_tensor() operation.
struct Exl3MtpTensorBinding {
    std::string_view name;
    std::filesystem::path shard_path;
    TensorInfo info;
};

struct Exl3MtpManifestSummary {
    std::size_t tensor_count = 0;
    std::uint64_t total_bytes = 0;
};

struct Exl3MtpBinding {
    std::vector<Exl3MtpTensorBinding> tensors;
    std::uint64_t total_bytes = 0;

    [[nodiscard]] const Exl3MtpTensorBinding* find(std::string_view name) const noexcept;
};

// The order is stable and is the order used by bind_qwen3_8_27b_native_mtp().
[[nodiscard]] std::span<const Exl3MtpTensorSpec>
qwen3_8_27b_native_mtp_manifest() noexcept;

// Validate only IndexedSafetensors headers.  This rejects missing, extra,
// duplicate, wrongly typed, wrongly shaped, wrongly sized, and malformed MTP
// entries; non-MTP model tensors are outside this manifest's scope.
[[nodiscard]] Exl3MtpManifestSummary validate_qwen3_8_27b_native_mtp(
    const IndexedSafetensors& collection);

// Validate and return a canonical metadata binding without materializing any
// tensor payload.  The returned descriptors may be passed to read_tensor().
[[nodiscard]] Exl3MtpBinding bind_qwen3_8_27b_native_mtp(
    const IndexedSafetensors& collection);

} // namespace ninfer::exl3
