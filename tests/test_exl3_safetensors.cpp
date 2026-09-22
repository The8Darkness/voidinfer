#include "exl3/safetensors.h"

#include <nlohmann/json.hpp>

#include <array>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using Json = nlohmann::json;
using ninfer::exl3::SafetensorsError;

void write_u64_le(std::ofstream& file, std::uint64_t value) {
    std::array<char, 8> bytes{};
    for (unsigned i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<char>((value >> (8u * i)) & 0xffu);
    }
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::filesystem::path write_fixture(const Json& header, std::uint64_t payload_bytes) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
                      ("ninfer-exl3-safetensors-" + std::to_string(stamp) + ".safetensors");
    const auto encoded = header.dump();
    std::ofstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot create safetensors fixture"); }
    write_u64_le(file, encoded.size());
    file.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
    for (std::uint64_t i = 0; i < payload_bytes; ++i) {
        const char value = static_cast<char>(i & 0xffu);
        file.write(&value, 1);
    }
    file.close();
    return path;
}

template <typename Function>
void expect_error(Function&& function, std::string_view label) {
    try {
        function();
    } catch (const SafetensorsError&) {
        return;
    }
    throw std::runtime_error(std::string(label) + " was accepted");
}

Json valid_header() {
    return {
        {"__metadata", {{"format", "pt"}}},
        {"model.layers.0.self_attn.q_proj.weight.trellis",
         {{"dtype", "I16"}, {"shape", {2, 2}}, {"data_offsets", {0, 8}}}},
        {"model.layers.0.self_attn.q_proj.weight.suh",
         {{"dtype", "F16"}, {"shape", {2}}, {"data_offsets", {8, 12}}}},
        {"model.layers.0.self_attn.q_proj.weight.mul1",
         {{"dtype", "I32"}, {"shape", {1}}, {"data_offsets", {12, 16}}}},
    };
}

void inspect_external_artifact_if_requested() {
    const char* root_value = std::getenv("NINFER_EXL3_TARGET_PATH");
    if (root_value == nullptr || *root_value == '\0') { return; }
    const std::filesystem::path root(root_value);
    const auto collection = ninfer::exl3::inspect_indexed_directory(root);
    if (collection.shards.size() != 3 || collection.indexed_tensor_count != 3080 ||
        collection.header_tensor_count != 3080) {
        throw std::runtime_error("external EXL3 target index/shard summary mismatch");
    }
    std::cout << "external_target_shards=" << collection.shards.size()
              << " external_target_tensors=" << collection.header_tensor_count << '\n';

    const char* drafter_value = std::getenv("NINFER_EXL3_DRAFTER_PATH");
    if (drafter_value == nullptr || *drafter_value == '\0') { return; }
    const auto drafter_header =
        ninfer::exl3::inspect_file(std::filesystem::path(drafter_value) / "model.safetensors");
    if (drafter_header.file_bytes == 0 || drafter_header.tensors.empty()) {
        throw std::runtime_error("invalid external EXL3 drafter");
    }
    std::cout << "external_drafter_tensors=" << drafter_header.tensors.size() << '\n';
}

void inspect_index_fixture() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path() /
                           ("ninfer-exl3-index-" + std::to_string(stamp));
    std::filesystem::create_directory(directory);
    try {
        const auto shard = directory / "model-00001-of-00001.safetensors";
        const auto encoded = valid_header().dump();
        std::ofstream file(shard, std::ios::binary);
        write_u64_le(file, encoded.size());
        file.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
        for (int i = 0; i < 16; ++i) { file.put(static_cast<char>(i)); }
        file.close();

        Json index = {
            {"metadata", {{"total_size", 16}}},
            {"weight_map",
             {{"model.layers.0.self_attn.q_proj.weight.trellis", shard.filename().string()},
              {"model.layers.0.self_attn.q_proj.weight.suh", shard.filename().string()},
              {"model.layers.0.self_attn.q_proj.weight.mul1", shard.filename().string()}}},
        };
        std::ofstream index_file(directory / "model.safetensors.index.json", std::ios::binary);
        index_file << index.dump();
        index_file.close();
        const auto collection = ninfer::exl3::inspect_indexed_directory(directory);
        if (collection.shards.size() != 1 || collection.indexed_tensor_count != 3 ||
            collection.header_tensor_count != 3 || !collection.has_index_total_size ||
            collection.index_total_size != 16) {
            throw std::runtime_error("indexed safetensors fixture mismatch");
        }

        index["weight_map"]["model.layers.0.self_attn.q_proj.weight.mul1"] = "missing.safetensors";
        std::ofstream bad_index(directory / "model.safetensors.index.json", std::ios::binary);
        bad_index << index.dump();
        bad_index.close();
        expect_error([&] { (void)ninfer::exl3::inspect_indexed_directory(directory); },
                     "missing indexed shard");
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }
    std::filesystem::remove_all(directory);
}

} // namespace

void inspect_dual_fixture_if_requested() {
    const char* root = std::getenv("NINFER_EXL3_MIRROR_UNIT_DIR");
    if (!root || !*root) return;
    const auto directory = std::filesystem::path(root);
    const auto manifest_path = directory / "manifest.json";
    Json manifest;
    { std::ifstream f(manifest_path); f >> manifest; }
    const auto primary = std::filesystem::path(manifest.at(0).at("c").get<std::string>());
    const auto collection = ninfer::exl3::inspect_indexed_directory(primary.parent_path());
    const auto sidecar = directory / ".ninfer" / "dual-load.json";
    std::filesystem::create_directories(sidecar.parent_path());
    std::filesystem::copy_file(manifest_path, sidecar, std::filesystem::copy_options::overwrite_existing);
    if (ninfer::exl3::find_verified_dual_manifest(directory) != sidecar)
        throw std::runtime_error("local default mirror discovery failed");
#ifdef _WIN32
    auto counterpart = directory.wstring(); counterpart[0] = L'C';
    if (ninfer::exl3::find_verified_dual_manifest(counterpart) != sidecar)
        throw std::runtime_error("absolute C/D default mirror discovery failed");
#endif
    if (!ninfer::exl3::find_verified_dual_manifest(directory / "missing-default").empty())
        throw std::runtime_error("unverified default mirror discovered");
    {
        ninfer::exl3::VerifiedDualTensorFiles reader(collection, manifest_path);
        std::uint64_t bytes = 0;
        for (const auto& shard : collection.shards) for (const auto& tensor : shard.header.tensors) {
            const auto ordinary = ninfer::exl3::read_tensor(shard.path, shard.header, tensor.name);
            std::vector<std::byte> guarded(ordinary.bytes().size() + 32, std::byte{0x5a});
            const auto output = std::span<std::byte>(guarded).subspan(16, ordinary.bytes().size());
            const auto half = output.size() / 2;
            auto first = std::async(std::launch::async, [&] { reader.read_into(tensor.name, 0, output.first(half), 0); });
            auto second = std::async(std::launch::async, [&] { reader.read_into(tensor.name, half, output.subspan(half), 1); });
            first.get(); second.get();
            if (!std::equal(output.begin(), output.end(), ordinary.bytes().begin()))
                throw std::runtime_error("parallel fixture byte mismatch");
            reader.read_into(tensor.name, 0, output, 0);
            if (!std::equal(output.begin(), output.end(), ordinary.bytes().begin()) ||
                !std::all_of(guarded.begin(), guarded.begin() + 16, [](std::byte b) { return b == std::byte{0x5a}; }) ||
                !std::all_of(guarded.end() - 16, guarded.end(), [](std::byte b) { return b == std::byte{0x5a}; }))
                throw std::runtime_error("parallel repeat/canary mismatch");
            reader.read_into(tensor.name, tensor.bytes(), {}, 1);
            expect_error([&] { reader.read_into(tensor.name, tensor.bytes() + 1, {}, 0); }, "offset beyond tensor");
            expect_error([&] { reader.read_into(tensor.name, tensor.bytes(), output.first(1), 0); }, "range beyond tensor");
            bytes += output.size();
        }
        if (reader.source_bytes(0) + reader.source_bytes(1) != 2 * bytes)
            throw std::runtime_error("parallel byte accounting mismatch");
        expect_error([&] { (void)reader.info("absent"); }, "absent parallel tensor");
        expect_error([&] { reader.read_into("absent", 0, {}, 0); }, "absent parallel range");
        expect_error([&] { (void)reader.source_bytes(2); }, "invalid parallel source");
        std::ofstream denied(primary, std::ios::binary | std::ios::app);
        if (denied) throw std::runtime_error("mirror allowed concurrent write access");
    }
    { std::ofstream allowed(primary, std::ios::binary | std::ios::app);
      if (!allowed) throw std::runtime_error("mirror retained lock after destruction"); }
    auto bad_collection = collection;
    bad_collection.shards[0].header.tensors[0].shape[0] += 1;
    expect_error([&] { ninfer::exl3::VerifiedDualTensorFiles bad_reader(bad_collection, manifest_path); }, "mismatched tensor metadata");
    const auto bad_path = directory / "bad-manifest.json";
    auto reject = [&](Json bad, std::string_view label) {
        { std::ofstream f(bad_path); f << bad.dump(); }
        expect_error([&] { ninfer::exl3::VerifiedDualTensorFiles bad_reader(collection, bad_path); }, label);
    };
    auto bad = manifest; bad[0]["bytes"] = bad[0]["bytes"].get<std::uint64_t>() + 1; reject(bad, "changed file size");
    bad = manifest; bad[0]["d_mtime_ns"] = bad[0]["d_mtime_ns"].get<std::uint64_t>() + 100; reject(bad, "changed mirror time");
    bad = manifest; bad.push_back(bad[0]); reject(bad, "duplicate mirror record");
    bad = manifest; bad[0]["sha256"] = "missing"; reject(bad, "invalid mirror identity");
    bad = manifest; bad[0]["d"] = "relative.safetensors"; reject(bad, "relative mirror path");
    bad = manifest; bad[0].erase("c_mtime_ns"); reject(bad, "missing mirror stamp");
    std::filesystem::remove(bad_path);
    std::cout << "parallel_safetensors_fixture: exact concurrent ranges, repeats, canaries, locks and manifest guards PASS\n";
}

int main() {
    const auto fixture = write_fixture(valid_header(), 16);
    try {
        const auto header = ninfer::exl3::inspect_file(fixture);
        if (header.tensors.size() != 3 || header.metadata_keys.size() != 1 ||
            header.payload_bytes != 16) {
            throw std::runtime_error("valid safetensors fixture summary mismatch");
        }
        const auto* trellis = header.find("model.layers.0.self_attn.q_proj.weight.trellis");
        if (trellis == nullptr || trellis->dtype != "I16" || trellis->bytes() != 8) {
            throw std::runtime_error("EXL3 trellis fixture mismatch");
        }

        auto bad_truncated = valid_header();
        const auto truncated = write_fixture(bad_truncated, 8);
        expect_error([&] { (void)ninfer::exl3::inspect_file(truncated); }, "truncated tensor range");
        std::filesystem::remove(truncated);

        auto bad_overlap = valid_header();
        bad_overlap["model.layers.0.self_attn.q_proj.weight.suh"]["data_offsets"] = {4, 8};
        const auto overlap = write_fixture(bad_overlap, 16);
        expect_error([&] { (void)ninfer::exl3::inspect_file(overlap); }, "overlapping tensors");
        std::filesystem::remove(overlap);

        auto bad_dtype_size = valid_header();
        bad_dtype_size["model.layers.0.self_attn.q_proj.weight.mul1"]["dtype"] = "I16";
        const auto dtype_size = write_fixture(bad_dtype_size, 16);
        expect_error([&] { (void)ninfer::exl3::inspect_file(dtype_size); }, "dtype size mismatch");
        std::filesystem::remove(dtype_size);

        inspect_index_fixture();
        inspect_dual_fixture_if_requested();
        inspect_external_artifact_if_requested();
        std::filesystem::remove(fixture);
        std::cout << "exl3_safetensors_test: ok\n";
    } catch (...) {
        std::filesystem::remove(fixture);
        throw;
    }
}
