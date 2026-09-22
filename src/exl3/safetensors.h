#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace ninfer::exl3 {

class SafetensorsError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct TensorInfo {
    std::string name;
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t data_begin = 0;
    std::uint64_t data_end   = 0;

    std::uint64_t bytes() const noexcept { return data_end - data_begin; }
};

// A read-only owned view of one safetensors payload.  The bytes are copied only for
// the explicitly requested tensor; the source file and its framing are never changed.
struct TensorPayload {
    TensorInfo info;
    std::shared_ptr<const std::vector<std::byte>> storage;

    std::span<const std::byte> bytes() const noexcept {
        return std::span<const std::byte>(*storage);
    }

    template <typename T>
    std::span<const T> typed(std::string_view expected_dtype) const {
        static_assert(std::is_trivially_copyable_v<T>);
        if (info.dtype != expected_dtype) {
            throw SafetensorsError("tensor " + info.name + " has dtype " + info.dtype +
                                   ", expected " + std::string(expected_dtype));
        }
        if (bytes().size_bytes() % sizeof(T) != 0) {
            throw SafetensorsError("tensor " + info.name + " payload is not T-aligned");
        }
        const auto* ptr = reinterpret_cast<const T*>(bytes().data());
        return std::span<const T>(ptr, bytes().size_bytes() / sizeof(T));
    }
};

struct SafetensorsHeader {
    std::uint64_t file_bytes    = 0;
    std::uint64_t header_bytes  = 0;
    std::uint64_t payload_bytes = 0;
    std::vector<std::string> metadata_keys;
    std::vector<TensorInfo> tensors;

    const TensorInfo* find(std::string_view name) const noexcept;
};

struct IndexedSafetensors {
    struct Shard {
        std::filesystem::path path;
        SafetensorsHeader header;
    };

    std::uint64_t index_total_size = 0;
    bool has_index_total_size      = false;
    std::size_t indexed_tensor_count = 0;
    std::size_t header_tensor_count  = 0;
    std::vector<Shard> shards;
};

// Read and validate only the safetensors framing/header. No tensor payload is loaded
// or modified by this function.
SafetensorsHeader inspect_file(const std::filesystem::path& path);

// Read exactly one already-validated tensor payload. This is intentionally explicit:
// callers cannot obtain an unvalidated range or an arbitrary file slice through this API.
TensorPayload read_tensor(const std::filesystem::path& path,
                          const SafetensorsHeader& header,
                          std::string_view name);

TensorPayload read_tensor(const std::filesystem::path& path, std::string_view name);

// Read and validate a Transformers-style model.safetensors.index.json and every
// referenced shard. No tensor payload is loaded or modified.
IndexedSafetensors inspect_indexed_directory(
    const std::filesystem::path& directory,
    std::string_view index_filename = "model.safetensors.index.json");

// Discover only a published verification sidecar, locally or at the absolute
// C/D counterpart directory. Absence does not authorize an unverified mirror.
std::filesystem::path find_verified_dual_manifest(const std::filesystem::path& directory);

// Explicitly consumes an externally SHA256-verified mirror manifest. File size and
// write timestamps are checked while read-only handles prevent replacement/writes.
// The caller supplies bounded staging buffers and one reader per C/D source.
class VerifiedDualTensorFiles {
public:
    VerifiedDualTensorFiles(const IndexedSafetensors& collection,
                         const std::filesystem::path& verified_manifest);
    ~VerifiedDualTensorFiles();
    VerifiedDualTensorFiles(const VerifiedDualTensorFiles&) = delete;
    VerifiedDualTensorFiles& operator=(const VerifiedDualTensorFiles&) = delete;
    TensorInfo info(std::string_view name) const;
    void read_into(std::string_view name, std::uint64_t offset,
                   std::span<std::byte> destination, int source) const;
    std::uint64_t source_bytes(int source) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::exl3
