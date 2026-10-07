// Research capture for the KV-tier (VeriCache) fidelity study. Prefills one
// real token stream and writes, per full-attention layer, the cached K
// (post-RoPE) and V rows for every position plus the query rows and the
// engine's attention output for the last `queries` positions. Not a test.
//
//   NINFER_EXL3_TARGET_PATH   model directory
//   KV_CAPTURE_IDS            whitespace-separated token-id files, concatenated
//   KV_CAPTURE_QUERIES        number of trailing query rows (default 32)
//   KV_CAPTURE_OUT            output directory
#include "exl3/text_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer::exl3;
constexpr int kKV = 4 * 256, kQ = 24 * 256;

void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }

struct Capture {
    int total = 0, queries = 0;
    std::vector<std::vector<std::uint16_t>> k, v, q, out;
    std::vector<int> layers;
};

void copy_rows(std::vector<std::uint16_t>& dst, std::size_t dst_row, const std::uint16_t* src,
               int src_row, int rows, int width, cudaStream_t stream) {
    cudaError_t e = cudaMemcpyAsync(dst.data() + dst_row * width,
        src + static_cast<std::size_t>(src_row) * width, static_cast<std::size_t>(rows) * width * 2,
        cudaMemcpyDeviceToHost, stream);
    require(e == cudaSuccess, "capture copy");
}

void observe(const Exl3LayerObservation& o, void* user) {
    auto& c = *static_cast<Capture*>(user);
    if (o.attention.k_rope == nullptr) return;
    int slot = -1;
    for (std::size_t i = 0; i < c.layers.size(); ++i) if (c.layers[i] == o.layer) slot = static_cast<int>(i);
    if (slot < 0) {
        slot = static_cast<int>(c.layers.size());
        c.layers.push_back(o.layer);
        c.k.emplace_back(static_cast<std::size_t>(c.total) * kKV);
        c.v.emplace_back(static_cast<std::size_t>(c.total) * kKV);
        c.q.emplace_back(static_cast<std::size_t>(c.queries) * kQ);
        c.out.emplace_back(static_cast<std::size_t>(c.queries) * kQ);
    }
    require(o.position + o.rows <= c.total, "capture position extent");
    copy_rows(c.k[slot], o.position, o.attention.k_rope, 0, o.rows, kKV, o.stream);
    copy_rows(c.v[slot], o.position, o.attention.v_projection, 0, o.rows, kKV, o.stream);
    const int first_query = c.total - c.queries;
    const int begin = std::max(first_query, o.position), end = o.position + o.rows;
    if (begin < end) {
        copy_rows(c.q[slot], begin - first_query, o.attention.q_rope, begin - o.position, end - begin, kQ, o.stream);
        copy_rows(c.out[slot], begin - first_query, o.attention.attention_output, begin - o.position,
                  end - begin, kQ, o.stream);
    }
    require(cudaStreamSynchronize(o.stream) == cudaSuccess, "capture synchronize");
}

void write(const std::filesystem::path& path, const std::vector<std::uint16_t>& data) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * 2));
    require(static_cast<bool>(f), "write " + path.string());
}
} // namespace

int main() {
    try {
        const char* target = std::getenv("NINFER_EXL3_TARGET_PATH");
        const char* ids = std::getenv("KV_CAPTURE_IDS");
        const char* out = std::getenv("KV_CAPTURE_OUT");
        if (!target || !ids || !out) { std::cerr << "set NINFER_EXL3_TARGET_PATH, KV_CAPTURE_IDS, KV_CAPTURE_OUT\n"; return 77; }
        std::vector<std::int64_t> tokens;
        std::stringstream files(ids);
        for (std::string file; std::getline(files, file, ';');) {
            std::ifstream f(file);
            require(static_cast<bool>(f), "open " + file);
            for (std::int64_t t; f >> t;) tokens.push_back(t);
        }
        Capture c;
        c.total = static_cast<int>(tokens.size());
        c.queries = std::getenv("KV_CAPTURE_QUERIES") ? std::atoi(std::getenv("KV_CAPTURE_QUERIES")) : 32;
        require(c.queries > 0 && c.queries < c.total, "query rows");
        auto model = Exl3TextModel::load(target, c.total + 16);
        auto context = model->create_context(false);
        context->set_layer_observer_for_test(&observe, &c);
        // Initial prefill is limited to 16 rows; the rest is teacher-forced in 8-row appends.
        const std::span<const std::int64_t> all(tokens.data(), tokens.size());
        context->prefill(all.first(16));
        for (std::size_t at = 16; at < all.size(); at += 8)
            context->append_prefill(all.subspan(at, std::min<std::size_t>(8, all.size() - at)));
        require(cudaDeviceSynchronize() == cudaSuccess, "synchronize prefill");
        const std::filesystem::path root(out);
        std::filesystem::create_directories(root);
        for (std::size_t i = 0; i < c.layers.size(); ++i) {
            const auto name = "L" + std::to_string(c.layers[i]);
            write(root / (name + "_k.bin"), c.k[i]);
            write(root / (name + "_v.bin"), c.v[i]);
            write(root / (name + "_q.bin"), c.q[i]);
            write(root / (name + "_out.bin"), c.out[i]);
        }
        std::ofstream meta(root / "meta.json");
        meta << "{\"tokens\":" << c.total << ",\"queries\":" << c.queries << ",\"layers\":[";
        for (std::size_t i = 0; i < c.layers.size(); ++i) meta << (i ? "," : "") << c.layers[i];
        meta << "]}\n";
        std::cout << "KV_CAPTURE tokens=" << c.total << " queries=" << c.queries << " layers=" << c.layers.size() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "KV_CAPTURE FAIL: " << e.what() << "\n";
        return 1;
    }
}
