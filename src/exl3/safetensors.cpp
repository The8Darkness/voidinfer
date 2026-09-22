#include "exl3/safetensors.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cwctype>
#include <fstream>
#include <functional>
#include <mutex>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ninfer::exl3 {
namespace {

using Json = nlohmann::json;

constexpr std::uint64_t kPrefixBytes    = 8;
constexpr std::uint64_t kMaxHeaderBytes = 128ull * 1024ull * 1024ull;

std::uint64_t read_u64_le(const std::array<unsigned char, 8>& bytes) noexcept {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < bytes.size(); ++i) {
        value |= std::uint64_t(bytes[i]) << (8u * i);
    }
    return value;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right, std::string_view label) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw SafetensorsError(std::string(label) + " overflows u64");
    }
    return left + right;
}

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right, std::string_view label) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw SafetensorsError(std::string(label) + " overflows u64");
    }
    return left * right;
}

Json read_json_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw SafetensorsError("cannot open JSON file: " + path.string()); }
    std::string encoded((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    try {
        return Json::parse(encoded);
    } catch (const Json::exception& exception) {
        throw SafetensorsError("invalid JSON in " + path.string() + ": " + exception.what());
    }
}

std::uint64_t dtype_bytes(std::string_view dtype) {
    if (dtype == "BOOL" || dtype == "U8" || dtype == "I8" || dtype == "F8_E4M3" ||
        dtype == "F8_E5M2") {
        return 1;
    }
    if (dtype == "U16" || dtype == "I16" || dtype == "F16" || dtype == "BF16") {
        return 2;
    }
    if (dtype == "U32" || dtype == "I32" || dtype == "F32") { return 4; }
    if (dtype == "U64" || dtype == "I64" || dtype == "F64") { return 8; }
    throw SafetensorsError("unsupported safetensors dtype: " + std::string(dtype));
}

const std::string& require_string(const Json& value, std::string_view label) {
    if (!value.is_string()) {
        throw SafetensorsError(std::string(label) + " must be a string");
    }
    return value.get_ref<const std::string&>();
}

std::uint64_t require_u64(const Json& value, std::string_view label) {
    if (!value.is_number_unsigned()) {
        throw SafetensorsError(std::string(label) + " must be a non-negative integer");
    }
    return value.get<std::uint64_t>();
}

TensorInfo parse_tensor(std::string_view name, const Json& value, std::uint64_t payload_bytes) {
    if (!value.is_object() || !value.contains("dtype") || !value.contains("shape") ||
        !value.contains("data_offsets")) {
        throw SafetensorsError("tensor " + std::string(name) + " has an invalid descriptor");
    }

    TensorInfo tensor;
    tensor.name  = std::string(name);
    tensor.dtype = require_string(value.at("dtype"), "tensor dtype");
    const auto element_bytes = dtype_bytes(tensor.dtype);

    const auto& shape = value.at("shape");
    if (!shape.is_array()) {
        throw SafetensorsError("tensor " + tensor.name + " shape must be an array");
    }
    tensor.shape.reserve(shape.size());
    std::uint64_t elements = 1;
    for (const auto& dimension : shape) {
        const auto dimension_value = require_u64(dimension, "tensor shape dimension");
        tensor.shape.push_back(dimension_value);
        elements = checked_mul(elements, dimension_value, "tensor element count");
    }

    const auto& offsets = value.at("data_offsets");
    if (!offsets.is_array() || offsets.size() != 2) {
        throw SafetensorsError("tensor " + tensor.name + " data_offsets must have two entries");
    }
    tensor.data_begin = require_u64(offsets.at(0), "tensor data begin");
    tensor.data_end   = require_u64(offsets.at(1), "tensor data end");
    if (tensor.data_end < tensor.data_begin || tensor.data_end > payload_bytes) {
        throw SafetensorsError("tensor " + tensor.name + " data range is outside the payload");
    }

    const auto expected_bytes = checked_mul(elements, element_bytes, "tensor byte count");
    if (tensor.bytes() != expected_bytes) {
        throw SafetensorsError("tensor " + tensor.name + " byte count does not match dtype/shape");
    }
    return tensor;
}

} // namespace

const TensorInfo* SafetensorsHeader::find(std::string_view name) const noexcept {
    for (const auto& tensor : tensors) {
        if (tensor.name == name) { return &tensor; }
    }
    return nullptr;
}

TensorPayload read_tensor(const std::filesystem::path& path,
                          const SafetensorsHeader& header,
                          std::string_view name) {
    const auto* info = header.find(name);
    if (info == nullptr) {
        throw SafetensorsError("tensor is absent from safetensors file: " + std::string(name));
    }

    const auto payload_start = checked_add(kPrefixBytes, header.header_bytes,
                                           "safetensors payload start");
    const auto file_begin = checked_add(payload_start, info->data_begin,
                                        "safetensors tensor file offset");
    const auto file_end = checked_add(payload_start, info->data_end,
                                      "safetensors tensor file end");
    if (file_end > header.file_bytes || file_begin > file_end) {
        throw SafetensorsError("tensor " + info->name + " is outside the validated file");
    }

    auto storage = std::make_shared<std::vector<std::byte>>(
        static_cast<std::size_t>(info->bytes()));
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw SafetensorsError("cannot open safetensors payload: " + path.string()); }
    if (file_begin > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        throw SafetensorsError("safetensors tensor offset exceeds stream range");
    }
    file.seekg(static_cast<std::streamoff>(file_begin), std::ios::beg);
    if (!file) { throw SafetensorsError("cannot seek to tensor " + info->name); }
    if (!storage->empty()) {
        file.read(reinterpret_cast<char*>(storage->data()),
                  static_cast<std::streamsize>(storage->size()));
        if (file.gcount() != static_cast<std::streamsize>(storage->size())) {
            throw SafetensorsError("cannot read complete tensor payload for " + info->name);
        }
    }

    return TensorPayload{*info, std::const_pointer_cast<const std::vector<std::byte>>(storage)};
}

TensorPayload read_tensor(const std::filesystem::path& path, std::string_view name) {
    return read_tensor(path, inspect_file(path), name);
}

bool safe_relative_filename(std::string_view value) {
    const std::filesystem::path path(value);
    if (path.empty() || path.is_absolute() || path.has_root_name() || path.has_root_directory()) {
        return false;
    }
    for (const auto& component : path) {
        if (component == "..") { return false; }
    }
    return true;
}

SafetensorsHeader inspect_file(const std::filesystem::path& path) {
    std::error_code error;
    const auto file_bytes = std::filesystem::file_size(path, error);
    if (error) {
        throw SafetensorsError("cannot stat safetensors file " + path.string() + ": " +
                               error.message());
    }
    if (file_bytes < kPrefixBytes) {
        throw SafetensorsError("safetensors file is shorter than its 8-byte prefix");
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) { throw SafetensorsError("cannot open safetensors file: " + path.string()); }

    std::array<unsigned char, 8> prefix{};
    file.read(reinterpret_cast<char*>(prefix.data()), static_cast<std::streamsize>(prefix.size()));
    if (file.gcount() != static_cast<std::streamsize>(prefix.size())) {
        throw SafetensorsError("cannot read safetensors header length");
    }

    const auto header_bytes = read_u64_le(prefix);
    if (header_bytes > kMaxHeaderBytes) {
        throw SafetensorsError("safetensors header exceeds the audit safety limit");
    }
    const auto payload_start = checked_add(kPrefixBytes, header_bytes, "safetensors payload start");
    if (payload_start > file_bytes) {
        throw SafetensorsError("safetensors header extends beyond the file");
    }

    std::string encoded_header(static_cast<std::size_t>(header_bytes), '\0');
    if (header_bytes != 0) {
        file.read(encoded_header.data(), static_cast<std::streamsize>(header_bytes));
        if (file.gcount() != static_cast<std::streamsize>(header_bytes)) {
            throw SafetensorsError("cannot read complete safetensors header");
        }
    }

    Json root;
    try {
        root = Json::parse(encoded_header);
    } catch (const Json::exception& exception) {
        throw SafetensorsError("invalid safetensors header JSON: " + std::string(exception.what()));
    }
    if (!root.is_object()) { throw SafetensorsError("safetensors header must be a JSON object"); }

    SafetensorsHeader result;
    result.file_bytes    = file_bytes;
    result.header_bytes  = header_bytes;
    result.payload_bytes = file_bytes - payload_start;

    for (const auto& [name, value] : root.items()) {
        // Safetensors reserves the standard spelling __metadata__. Accept the
        // historical one-underscore spelling too for compatibility with older
        // local fixtures, but never treat either as a tensor descriptor.
        if (name == "__metadata__" || name == "__metadata") {
            if (!value.is_object()) { throw SafetensorsError("__metadata must be an object"); }
            for (const auto& [metadata_name, metadata_value] : value.items()) {
                if (!metadata_value.is_string()) {
                    throw SafetensorsError("safetensors metadata values must be strings");
                }
                result.metadata_keys.push_back(metadata_name);
            }
            continue;
        }
        result.tensors.push_back(parse_tensor(name, value, result.payload_bytes));
    }

    std::vector<const TensorInfo*> by_offset;
    by_offset.reserve(result.tensors.size());
    for (const auto& tensor : result.tensors) { by_offset.push_back(&tensor); }
    std::sort(by_offset.begin(), by_offset.end(), [](const auto* left, const auto* right) {
        if (left->data_begin != right->data_begin) { return left->data_begin < right->data_begin; }
        return left->data_end < right->data_end;
    });
    std::uint64_t previous_end = 0;
    for (const auto* tensor : by_offset) {
        if (tensor->data_begin < previous_end) {
            throw SafetensorsError("safetensors tensor data ranges overlap");
        }
        previous_end = std::max(previous_end, tensor->data_end);
    }
    return result;
}

IndexedSafetensors inspect_indexed_directory(const std::filesystem::path& directory,
                                             std::string_view index_filename) {
    if (!safe_relative_filename(index_filename)) {
        throw SafetensorsError("safetensors index filename must be a safe relative path");
    }
    const auto index_path = directory / std::filesystem::path(index_filename);
    const auto root       = read_json_file(index_path);
    if (!root.is_object() || !root.contains("weight_map") || !root.at("weight_map").is_object()) {
        throw SafetensorsError("safetensors index must contain an object weight_map");
    }

    IndexedSafetensors result;
    if (root.contains("metadata")) {
        const auto& metadata = root.at("metadata");
        if (!metadata.is_object()) { throw SafetensorsError("safetensors index metadata must be an object"); }
        if (metadata.contains("total_size")) {
            result.index_total_size = require_u64(metadata.at("total_size"), "index total_size");
            result.has_index_total_size = true;
        }
    }

    std::map<std::string, std::vector<std::string>> names_by_shard;
    for (const auto& [tensor_name, shard_value] : root.at("weight_map").items()) {
        const auto& shard_name = require_string(shard_value, "weight_map shard filename");
        if (!safe_relative_filename(shard_name)) {
            throw SafetensorsError("weight_map contains an unsafe shard filename");
        }
        names_by_shard[shard_name].push_back(tensor_name);
    }
    if (names_by_shard.empty()) { throw SafetensorsError("safetensors weight_map is empty"); }

    for (const auto& [shard_name, indexed_names] : names_by_shard) {
        const auto shard_path = directory / std::filesystem::path(shard_name);
        auto header            = inspect_file(shard_path);
        std::unordered_set<std::string> indexed_set(indexed_names.begin(), indexed_names.end());
        for (const auto& name : indexed_names) {
            if (header.find(name) == nullptr) {
                throw SafetensorsError("index tensor " + name + " is absent from shard " + shard_name);
            }
        }
        if (indexed_set.size() != header.tensors.size()) {
            throw SafetensorsError("index has duplicate tensor names in shard " + shard_name);
        }
        for (const auto& tensor : header.tensors) {
            if (!indexed_set.contains(tensor.name)) {
                throw SafetensorsError("shard tensor " + tensor.name + " is absent from the index");
            }
        }
        result.indexed_tensor_count += indexed_names.size();
        result.header_tensor_count += header.tensors.size();
        result.shards.push_back({shard_path, std::move(header)});
    }

    if (result.has_index_total_size) {
        std::uint64_t actual_total_size = 0;
        for (const auto& shard : result.shards) {
            for (const auto& tensor : shard.header.tensors) {
                actual_total_size = checked_add(actual_total_size, tensor.bytes(), "indexed tensor bytes");
            }
        }
        if (actual_total_size != result.index_total_size) {
            throw SafetensorsError("index total_size does not match shard tensor bytes");
        }
    }
    return result;
}

struct VerifiedDualTensorFiles::Impl {
    std::unordered_map<std::string, TensorInfo> tensors;
    std::unordered_map<std::string, std::function<void(int, std::uint64_t, std::span<std::byte>)>> readers;
    std::vector<std::shared_ptr<const void>> source_locks;
    mutable std::array<std::atomic<std::uint64_t>, 2> bytes{};
};

std::filesystem::path find_verified_dual_manifest(const std::filesystem::path& directory) {
    auto primary = std::filesystem::absolute(directory).lexically_normal();
    const auto local = primary / ".ninfer" / "dual-load.json";
    if (std::filesystem::exists(local)) return local;
#ifdef _WIN32
    const auto drive = primary.root_name().wstring();
    if (drive.size() == 2 && (std::towupper(drive[0]) == L'C' || std::towupper(drive[0]) == L'D')) {
        auto counterpart = primary.wstring();
        counterpart[0] = std::towupper(drive[0]) == L'C' ? L'D' : L'C';
        const auto other = std::filesystem::path(counterpart) / ".ninfer" / "dual-load.json";
        if (std::filesystem::exists(other)) return other;
    }
#endif
    return {};
}

#ifdef _WIN32
namespace {
struct ReadOnlyMirrorFile {
    HANDLE file = INVALID_HANDLE_VALUE;
    std::mutex mutex;
    ~ReadOnlyMirrorFile() { if (file != INVALID_HANDLE_VALUE) CloseHandle(file); }
    void open(const std::filesystem::path& path, std::uint64_t bytes, std::uint64_t mtime_ns) {
        file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) throw SafetensorsError("cannot lock mirror: " + path.string());
        BY_HANDLE_FILE_INFORMATION info{};
        if (!GetFileInformationByHandle(file, &info)) throw SafetensorsError("cannot stat locked mirror");
        const auto actual_bytes = (std::uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
        const auto actual_time = (std::uint64_t(info.ftLastWriteTime.dwHighDateTime) << 32) |
                                  info.ftLastWriteTime.dwLowDateTime;
        if (actual_bytes != bytes || actual_time != mtime_ns / 100 + 116444736000000000ull)
            throw SafetensorsError("verified mirror stamp changed: " + path.string());
    }
    void read(std::uint64_t offset, std::span<std::byte> destination) {
        std::lock_guard<std::mutex> guard(mutex);
        if (offset > static_cast<std::uint64_t>(std::numeric_limits<LONGLONG>::max()))
            throw SafetensorsError("mirror offset exceeds Windows file range");
        LARGE_INTEGER position{}; position.QuadPart = static_cast<LONGLONG>(offset);
        if (!SetFilePointerEx(file, position, nullptr, FILE_BEGIN)) throw SafetensorsError("cannot seek mirror");
        std::size_t done = 0;
        while (done < destination.size()) {
            const auto request = static_cast<DWORD>(std::min<std::size_t>(destination.size() - done, 32u * 1024u * 1024u));
            DWORD got = 0;
            if (!ReadFile(file, destination.data() + done, request, &got, nullptr) || got == 0)
                throw SafetensorsError("incomplete mirror read");
            done += got;
        }
    }
};
std::wstring normalized_mirror_path(const std::filesystem::path& path) {
    auto value = std::filesystem::absolute(path).lexically_normal().wstring();
    std::transform(value.begin(), value.end(), value.begin(), [](wchar_t c) { return std::towlower(c); });
    return value;
}
void match_mirror_header(const SafetensorsHeader& a, const SafetensorsHeader& b) {
    if (a.file_bytes != b.file_bytes || a.header_bytes != b.header_bytes ||
        a.payload_bytes != b.payload_bytes || a.tensors.size() != b.tensors.size())
        throw SafetensorsError("mirror framing mismatch");
    for (const auto& t : a.tensors) {
        const auto* other = b.find(t.name);
        if (!other || t.dtype != other->dtype || t.shape != other->shape ||
            t.data_begin != other->data_begin || t.data_end != other->data_end)
            throw SafetensorsError("mirror tensor metadata mismatch: " + t.name);
    }
}
} // namespace
#endif

VerifiedDualTensorFiles::VerifiedDualTensorFiles(const IndexedSafetensors& collection,
                                                 const std::filesystem::path& verified_manifest)
    : impl_(std::make_unique<Impl>()) {
#ifdef _WIN32
    try {
        const auto manifest = read_json_file(verified_manifest);
        if (!manifest.is_array() || collection.shards.empty()) throw SafetensorsError("invalid mirror manifest");
        for (const auto& shard : collection.shards) {
            const Json* record = nullptr;
            for (const auto& entry : manifest) {
                if (!entry.is_object() || !entry.contains("c") || !entry.contains("d"))
                    throw SafetensorsError("invalid mirror manifest entry");
                if (normalized_mirror_path(shard.path) == normalized_mirror_path(entry.at("c").get<std::string>()) ||
                    normalized_mirror_path(shard.path) == normalized_mirror_path(entry.at("d").get<std::string>())) {
                    if (record) throw SafetensorsError("duplicate mirror manifest entry");
                    record = &entry;
                }
            }
            if (!record) throw SafetensorsError("shard is not in verified mirror manifest");
            const auto hash = record->at("sha256").get<std::string>();
            if (hash.size() != 64 || !std::all_of(hash.begin(), hash.end(), [](char c) {
                    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f'); }))
                throw SafetensorsError("mirror manifest needs a verified SHA256 identity");
            std::array<std::shared_ptr<ReadOnlyMirrorFile>, 2> files;
            for (int source = 0; source < 2; ++source) {
                const char* key = source == 0 ? "c" : "d";
                const auto path = std::filesystem::path(record->at(key).get<std::string>());
                const auto drive = path.root_name().wstring();
                if (!path.is_absolute() || drive.size() != 2 || std::towupper(drive[0]) != (source == 0 ? L'C' : L'D'))
                    throw SafetensorsError("mirror source must be an absolute C/D path");
                files[source] = std::make_shared<ReadOnlyMirrorFile>();
                files[source]->open(path, record->at("bytes").get<std::uint64_t>(),
                                    record->at(std::string(key) + "_mtime_ns").get<std::uint64_t>());
                match_mirror_header(shard.header, inspect_file(path));
                impl_->source_locks.push_back(files[source]);
            }
            for (const auto& tensor : shard.header.tensors) {
                const auto base = checked_add(8 + shard.header.header_bytes, tensor.data_begin, "mirror tensor offset");
                const auto size = tensor.bytes();
                if (!impl_->tensors.emplace(tensor.name, tensor).second) throw SafetensorsError("duplicate mirror tensor name");
                impl_->readers.emplace(tensor.name, [files, base, size](int source, std::uint64_t offset, std::span<std::byte> output) {
                    if (offset > size || output.size() > size - offset) throw SafetensorsError("mirror read outside tensor");
                    files[source]->read(checked_add(base, offset, "mirror range offset"), output);
                });
            }
        }
    } catch (const Json::exception& error) {
        throw SafetensorsError(std::string("invalid verified mirror manifest: ") + error.what());
    }
#else
    (void)collection; (void)verified_manifest;
    throw SafetensorsError("C/D mirrored loading requires Windows");
#endif
}
VerifiedDualTensorFiles::~VerifiedDualTensorFiles() = default;
TensorInfo VerifiedDualTensorFiles::info(std::string_view name) const {
    const auto it = impl_->tensors.find(std::string(name));
    if (it == impl_->tensors.end()) throw SafetensorsError("tensor absent from mirror collection");
    return it->second;
}
void VerifiedDualTensorFiles::read_into(std::string_view name, std::uint64_t offset,
                                       std::span<std::byte> destination, int source) const {
    if (source < 0 || source > 1) throw SafetensorsError("invalid mirror source index");
    const auto it = impl_->readers.find(std::string(name));
    if (it == impl_->readers.end()) throw SafetensorsError("tensor absent from mirror collection");
    it->second(source, offset, destination);
    impl_->bytes[source].fetch_add(destination.size(), std::memory_order_relaxed);
}
std::uint64_t VerifiedDualTensorFiles::source_bytes(int source) const {
    if (source < 0 || source > 1) throw SafetensorsError("invalid mirror source index");
    return impl_->bytes[source].load(std::memory_order_relaxed);
}

} // namespace ninfer::exl3
