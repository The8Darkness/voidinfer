#include "exl3/linear_cuda.h"
#include "exl3/safetensors.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
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

void check(cudaError_t e, const char* op) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(e));
}
const char* env(const char* name) { const char* value = std::getenv(name); return value ? value : ""; }

struct Buffer {
    void* ptr = nullptr;
    ~Buffer() { if (ptr) cudaFree(ptr); }
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

TensorPayload load(const IndexedSafetensors& c, const std::string& name) {
    for (const auto& shard : c.shards) if (shard.header.find(name)) return read_tensor(shard.path, shard.header, name);
    throw std::runtime_error("missing tensor: " + name);
}

struct Linear {
    std::vector<TensorPayload> host;
    std::vector<std::unique_ptr<Buffer>> device;
    Exl3CudaLinearWeights view{};
    Exl3CudaLinearMetadata metadata{};
};

Linear make_linear(const IndexedSafetensors& c, const std::string& prefix) {
    Linear r;
    for (const char* suffix : {"trellis", "suh", "svh", "mul1"}) r.host.push_back(load(c, prefix + "." + suffix));
    auto add = [&](const TensorPayload& t) {
        auto b = std::make_unique<Buffer>();
        check(cudaMalloc(&b->ptr, t.bytes().size_bytes()), "allocate benchmark tensor");
        check(cudaMemcpy(b->ptr, t.bytes().data(), t.bytes().size_bytes(), cudaMemcpyHostToDevice), "upload benchmark tensor");
        void* p = b->ptr; r.device.push_back(std::move(b)); return p;
    };
    r.view.trellis = static_cast<const std::uint16_t*>(add(r.host[0]));
    r.view.suh = static_cast<const std::uint16_t*>(add(r.host[1]));
    r.view.svh = static_cast<const std::uint16_t*>(add(r.host[2]));
    r.view.mul1 = static_cast<const std::int32_t*>(add(r.host[3]));
    const int in = static_cast<int>(r.host[0].info.shape[0] * 16);
    const int out = static_cast<int>(r.host[0].info.shape[1] * 16);
    const int bits = static_cast<int>(r.host[0].info.shape[2] / 16);
    r.metadata = {in, out, bits, false, true, false};
    return r;
}

double benchmark(Exl3CudaLinearWorkspace& workspace, const Linear& linear,
                 const std::uint16_t* input, std::uint16_t* output) {
    const int warmup = env("NINFER_EXL3_WARMUP")[0] ? std::stoi(env("NINFER_EXL3_WARMUP")) : 15;
    const int iterations = env("NINFER_EXL3_ITERATIONS")[0] ? std::stoi(env("NINFER_EXL3_ITERATIONS")) : 60;
    for (int i = 0; i < warmup; ++i) workspace.forward(linear.view, linear.metadata, input, output, 1);
    check(cudaDeviceSynchronize(), "benchmark warmup synchronize");
    cudaEvent_t start{}, stop{}; check(cudaEventCreate(&start), "create start event"); check(cudaEventCreate(&stop), "create stop event");
    check(cudaEventRecord(start), "record start event");
    for (int i = 0; i < iterations; ++i) workspace.forward(linear.view, linear.metadata, input, output, 1);
    check(cudaEventRecord(stop), "record stop event"); check(cudaEventSynchronize(stop), "synchronize stop event");
    float ms = 0.0f; check(cudaEventElapsedTime(&ms, start, stop), "read benchmark events");
    cudaEventDestroy(start); cudaEventDestroy(stop); return static_cast<double>(ms) * 1000.0 / iterations;
}
} // namespace

int main() {
    try {
        const std::string target = env("NINFER_EXL3_TARGET_PATH");
        if (target.empty()) { std::cerr << "SKIP: set NINFER_EXL3_TARGET_PATH\n"; return 77; }
        const auto collection = inspect_indexed_directory(std::filesystem::path(target));
        const std::vector<std::pair<std::string, std::string>> modules = {
            {"h6_lm_head", "lm_head"},
            {"full_q_k5", "model.language_model.layers.3.self_attn.q_proj"},
            {"full_q", "model.language_model.layers.19.self_attn.q_proj"},
            {"full_k_k7", "model.language_model.layers.3.self_attn.k_proj"},
            {"full_k", "model.language_model.layers.19.self_attn.k_proj"},
            {"full_o_k7", "model.language_model.layers.23.self_attn.o_proj"},
            {"full_v", "model.language_model.layers.19.self_attn.v_proj"},
            {"full_o", "model.language_model.layers.19.self_attn.o_proj"},
            {"full_gate_k7", "model.language_model.layers.63.mlp.gate_proj"},
            {"full_gate", "model.language_model.layers.19.mlp.gate_proj"},
            {"full_up", "model.language_model.layers.19.mlp.up_proj"},
            {"full_down", "model.language_model.layers.19.mlp.down_proj"},
            {"gdn_qkv_k6", "model.language_model.layers.0.linear_attn.in_proj_qkv"},
            {"gdn_qkv", "model.language_model.layers.5.linear_attn.in_proj_qkv"},
            {"gdn_z_k5", "model.language_model.layers.1.linear_attn.in_proj_z"},
            {"gdn_z", "model.language_model.layers.5.linear_attn.in_proj_z"},
            {"gdn_o_k7", "model.language_model.layers.2.linear_attn.out_proj"},
            {"gdn_o", "model.language_model.layers.5.linear_attn.out_proj"},
            {"gdn_gate", "model.language_model.layers.5.mlp.gate_proj"},
            {"gdn_up", "model.language_model.layers.5.mlp.up_proj"},
            {"gdn_down_k5", "model.language_model.layers.0.mlp.down_proj"},
            {"gdn_down_k7", "model.language_model.layers.23.mlp.down_proj"},
            {"gdn_down", "model.language_model.layers.5.mlp.down_proj"}};
        for (const auto& [label, prefix] : modules) {
            const std::string only = env("NINFER_EXL3_BENCH_ONLY");
            if (!only.empty() && only != label) continue;
            auto linear = make_linear(collection, prefix);
            Buffer input, output;
            const int in = linear.metadata.in_features, out = linear.metadata.out_features;
            check(cudaMalloc(&input.ptr, static_cast<std::size_t>(in) * 2), "allocate benchmark input");
            check(cudaMalloc(&output.ptr, static_cast<std::size_t>(out) * 2), "allocate benchmark output");
            check(cudaMemset(input.ptr, 0, static_cast<std::size_t>(in) * 2), "clear benchmark input");
            Exl3CudaLinearWorkspace workspace(in, out, 1);
            const double us = benchmark(workspace, linear, static_cast<const std::uint16_t*>(input.ptr), static_cast<std::uint16_t*>(output.ptr));
            std::cout << "BENCH " << label << " input=" << in << " output=" << out << " K=" << linear.metadata.K
                      << " us=" << std::setprecision(10) << us << " launches=3\n";
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
