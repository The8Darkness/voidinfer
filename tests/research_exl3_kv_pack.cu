// Round trip of the lossless L0 KV page packer (src/exl3/l0_kv_pack.cuh) on
// captured FP16 K/V of the 16 full-attention layers: rows get real L0 OSCAR
// INT4 codes (l0_oscar::encode_kernel), every 64-row page above the sink is
// packed against them, decoded into a second plane set and compared bit for
// bit; a corrupted blob must be rejected.
//
// VOIDINFER_KV_PACK_DATA: directory with L{3+4b}_{k,v}.bin ([tokens][4][256]
// FP16, b = 0..15) and VOIDINFER_EXL3_L0_OSCAR_ROT: the rotation assets.
// Exit 77 (skip) without them.
#include "exl3/l0_kv_pack.cuh"
#include "exl3/l0_oscar.cuh"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace pack = voidinfer::exl3::l0_kv_pack;
namespace oscar = voidinfer::exl3::l0_oscar;

// The packer reads the rotations through rotation_view() (defined by the
// engine's L0 attention unit; this test links no engine library).
voidinfer::exl3::l0_oscar::RotationView voidinfer::exl3::l0_oscar::rotation_view() {
    const auto& a = assets();
    return {a.rk, a.rv, a.mu};
}

static void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
        std::exit(1);
    }
}

int main() {
    const char* data = std::getenv("VOIDINFER_KV_PACK_DATA");
    const char* rot = std::getenv("VOIDINFER_EXL3_L0_OSCAR_ROT");
    if (!data || !rot || !std::filesystem::exists(std::filesystem::path(data) / "L3_k.bin")) {
        std::printf("SKIP: set VOIDINFER_KV_PACK_DATA and VOIDINFER_EXL3_L0_OSCAR_ROT\n");
        return 77;
    }
    const int rows_env = std::getenv("VOIDINFER_KV_PACK_ROWS") ? std::atoi(std::getenv("VOIDINFER_KV_PACK_ROWS")) : 8256;
    const int rows = rows_env / 64 * 64;
    constexpr int banks = 16, row_words = 1024;
    const std::size_t plane_words = static_cast<std::size_t>(rows) * row_words;
    oscar::set_recent_rows();
    const auto& assets = oscar::assets();

    // Source planes (mapped pinned, as the L0 FP16 planes) and their codes.
    pack::Planes source{}, target{};
    std::vector<std::uint8_t*> chunk(banks);
    const std::size_t code_row = 4 * oscar::kCodeBytes, meta_row = 4 * oscar::kMetaFloats;
    std::uint8_t *k_codes, *v_codes;
    __half *k_meta, *v_meta;
    int* state;
    check(cudaMalloc(&k_codes, rows * code_row), "codes");
    check(cudaMalloc(&v_codes, rows * code_row), "codes");
    check(cudaMalloc(&k_meta, rows * meta_row * 2), "meta");
    check(cudaMalloc(&v_meta, rows * meta_row * 2), "meta");
    check(cudaMalloc(&state, 2 * sizeof(int)), "state");
    for (int b = 0; b < banks; ++b) {
        for (int plane = 0; plane < 2; ++plane) {
            std::uint16_t *src, *dst;
            check(cudaHostAlloc(&src, plane_words * 2, cudaHostAllocMapped | cudaHostAllocPortable), "planes");
            check(cudaHostAlloc(&dst, plane_words * 2, cudaHostAllocMapped | cudaHostAllocPortable), "planes");
            const auto path = std::filesystem::path(data) / ("L" + std::to_string(3 + 4 * b) + (plane ? "_v.bin" : "_k.bin"));
            std::ifstream f(path, std::ios::binary);
            f.read(reinterpret_cast<char*>(src), static_cast<std::streamsize>(plane_words * 2));
            if (!f) { std::fprintf(stderr, "read %s\n", path.string().c_str()); return 1; }
            std::memset(dst, 0xA5, plane_words * 2);
            (plane ? source.v : source.k)[b] = src;
            (plane ? target.v : target.k)[b] = dst;
        }
        const int init[2] = {oscar::kSink, 0};
        check(cudaMemcpy(state, init, sizeof(init), cudaMemcpyHostToDevice), "state");
        oscar::encode_kernel<<<oscar::kEncodeBlocks, 256>>>(source.k[b], source.v[b], rows, nullptr,
            rows + oscar::recent_rows() + 64, state, assets, b, k_codes, v_codes, k_meta, v_meta, oscar::HotView{});
        check(cudaDeviceSynchronize(), "encode");
        // Chunk layout of Exl3ExactHostState::L0CodeChunk: K codes, V codes, K meta, V meta.
        check(cudaHostAlloc(&chunk[b], rows * (2 * code_row + 4 * meta_row), cudaHostAllocPortable), "chunk");
        check(cudaMemcpy(chunk[b], k_codes, rows * code_row, cudaMemcpyDeviceToHost), "chunk");
        check(cudaMemcpy(chunk[b] + rows * code_row, v_codes, rows * code_row, cudaMemcpyDeviceToHost), "chunk");
        check(cudaMemcpy(chunk[b] + 2 * rows * code_row, k_meta, rows * meta_row * 2, cudaMemcpyDeviceToHost), "chunk");
        check(cudaMemcpy(chunk[b] + 2 * rows * code_row + rows * meta_row * 2, v_meta, rows * meta_row * 2,
            cudaMemcpyDeviceToHost), "chunk");
    }
    std::vector<pack::Packer::Page> pages;
    for (int row = oscar::kSink; row < rows; row += 64) {
        pack::Packer::Page page;
        page.row = row;
        page.rows = 64;
        for (int b = 0; b < banks; ++b) {
            page.k_codes[b] = chunk[b] + row * code_row;
            page.v_codes[b] = chunk[b] + rows * code_row + row * code_row;
            const auto* meta = reinterpret_cast<const std::uint16_t*>(chunk[b] + 2 * rows * code_row);
            page.k_meta[b] = meta + row * meta_row;
            page.v_meta[b] = meta + rows * meta_row + row * meta_row;
        }
        pages.push_back(page);
    }
    auto& packer = pack::Packer::instance();
    std::vector<std::vector<std::uint8_t>> blobs;
    packer.pack(source, banks, {pages.front()}, blobs);   // warm-up (allocations)
    auto started = std::chrono::steady_clock::now();
    packer.pack(source, banks, pages, blobs);
    const double pack_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    std::size_t packed = 0, raw_pages = 0, raw_streams = 0;
    for (const auto& blob : blobs) {
        if (blob.empty()) { ++raw_pages; continue; }
        packed += blob.size();
        const auto* entries = reinterpret_cast<const pack::Entry*>(blob.data() + sizeof(pack::Header));
        for (int s = 0; s < banks * 8; ++s) raw_streams += entries[s].mode;
    }
    const std::size_t raw_bytes = (pages.size() - raw_pages) * 64ull * row_words * 2 * 2 * banks;
    std::printf("pages %zu (raw %zu, raw streams %zu): %.1f MiB -> %.1f MiB = %.3fx, %.3f bits/value, pack %.1f ms (%.2f ms/page)\n",
        pages.size(), raw_pages, raw_streams, raw_bytes / 1048576.0, packed / 1048576.0,
        static_cast<double>(raw_bytes) / packed, 16.0 * packed / raw_bytes, pack_ms, pack_ms / pages.size());
    std::vector<pack::Packer::Page> coded;
    std::vector<const std::uint8_t*> pointers;
    std::vector<std::size_t> sizes;
    for (std::size_t p = 0; p < pages.size(); ++p)
        if (!blobs[p].empty()) { coded.push_back(pages[p]); pointers.push_back(blobs[p].data()); sizes.push_back(blobs[p].size()); }
    started = std::chrono::steady_clock::now();
    packer.unpack(&target, banks, coded, pointers, sizes);
    const double unpack_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    std::size_t mismatches = 0;
    for (const auto& page : coded)
        for (int b = 0; b < banks; ++b)
            for (int plane = 0; plane < 2; ++plane) {
                const auto* a = (plane ? source.v : source.k)[b] + static_cast<std::size_t>(page.row) * row_words;
                const auto* c = (plane ? target.v : target.k)[b] + static_cast<std::size_t>(page.row) * row_words;
                if (std::memcmp(a, c, static_cast<std::size_t>(page.rows) * row_words * 2) != 0) ++mismatches;
            }
    std::printf("unpack %.1f ms (%.2f ms/page), mismatching page planes %zu\n", unpack_ms, unpack_ms / coded.size(),
        mismatches);
    // Host-buffer destination (materializing a page's FP16 payload).
    {
        const std::size_t count = std::min<std::size_t>(5, coded.size());
        std::vector<std::vector<std::uint16_t>> host(count * banks * 2,
            std::vector<std::uint16_t>(static_cast<std::size_t>(64) * row_words));
        packer.unpack(nullptr, banks, {coded.begin(), coded.begin() + count}, {pointers.begin(), pointers.begin() + count},
            {sizes.begin(), sizes.begin() + count}, [&](std::size_t page, int bank, int plane) {
                return host[(page * banks + bank) * 2 + plane].data();
            });
        for (std::size_t page = 0; page < count; ++page)
            for (int b = 0; b < banks; ++b)
                for (int plane = 0; plane < 2; ++plane)
                    if (std::memcmp(host[(page * banks + b) * 2 + plane].data(),
                            (plane ? source.v : source.k)[b] + static_cast<std::size_t>(coded[page].row) * row_words,
                            static_cast<std::size_t>(64) * row_words * 2) != 0) ++mismatches;
        std::printf("host-buffer unpack of %zu pages, mismatching page planes %zu\n", count, mismatches);
    }
    // A flipped payload bit must be caught by the checksums.
    const std::size_t middle = coded.size() / 2;
    std::vector<std::uint8_t> corrupt(pointers[middle], pointers[middle] + sizes[middle]);
    corrupt[corrupt.size() / 2] ^= 0x10;
    bool rejected = false;
    try {
        const std::uint8_t* p = corrupt.data();
        packer.unpack(&target, banks, {coded[middle]}, {p}, {corrupt.size()});
    } catch (const std::exception& error) {
        rejected = true;
        std::printf("corrupted blob rejected: %s\n", error.what());
    }
    if (mismatches || !rejected || raw_pages == pages.size()) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
