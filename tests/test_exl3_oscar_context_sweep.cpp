// E4C1 Stage 11: same-build long-context A/B (ordinary F16 vs OSCAR eager).
#include "exl3/text_model.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

using namespace ninfer::exl3;

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

std::string env(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
}

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(operation);
}

double median_ms(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

std::uint64_t vram_used_bytes() {
    std::size_t free_bytes = 0, total_bytes = 0;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "sweep VRAM query");
    return static_cast<std::uint64_t>(total_bytes - free_bytes);
}

// Deterministic prompt of exactly length tokens.
std::vector<std::int64_t> make_prompt(int length) {
    static const std::int64_t base[] = {248045, 846, 198, 33963, 799, 2716, 2029, 883,
                                        47503, 13, 248046, 198, 248045, 74455};
    std::vector<std::int64_t> prompt;
    prompt.reserve(length);
    std::uint64_t lcg = 0xC10CULL;
    for (int i = 0; i < length; ++i) {
        if (i < 256) {
            prompt.push_back(base[i % 14]);
        } else {
            lcg = 1664525ULL * lcg + 1013904223ULL;
            prompt.push_back(static_cast<std::int64_t>((lcg >> 16U) % 248320U));
        }
    }
    return prompt;
}

struct SweepRow {
    int context = 0;
    bool oscar = false;
    std::int64_t greedy = -1;
    double prefill_ms = 0.0;
    double prefill_tok_s = 0.0;
    double decode_ms = 0.0;
    double tok_s = 0.0;
    std::uint64_t vram_bytes = 0;
    int split_class = 0;
};

SweepRow run_point(Exl3TextModel& model, int context, bool oscar, int decode_tokens) {
    SweepRow row;
    row.context = context;
    row.oscar = oscar;
    auto ctx = model.create_context(false);
    if (oscar) require(ctx->try_enable_oscar_from_environment(), "sweep OSCAR enable failed");
    // Leave room for timed decodes + greedy spot-check inside max_context.
    const int ingest = context - (decode_tokens + 2);
    require(ingest > 16, "sweep context too small for decode tail");
    const auto prompt = make_prompt(ingest);
    // EXL3 layers take at most 16 rows per forward; chunk long prefills.
    const auto pre0 = std::chrono::steady_clock::now();
    for (std::size_t off = 0; off < prompt.size(); off += 16) {
        const std::size_t count = std::min<std::size_t>(16, prompt.size() - off);
        if (off == 0) {
            ctx->prefill(std::span<const std::int64_t>(prompt.data(), count));
        } else {
            for (std::size_t i = 0; i < count; ++i) {
                ctx->decode(prompt[off + i]);
            }
        }
    }
    cudaDeviceSynchronize();
    const auto pre1 = std::chrono::steady_clock::now();
    row.prefill_ms = std::chrono::duration<double, std::milli>(pre1 - pre0).count();
    row.prefill_tok_s = 1000.0 * context / row.prefill_ms;
    row.vram_bytes = vram_used_bytes();
    std::vector<double> samples;
    std::int64_t token = 846;
    for (int i = 0; i < decode_tokens + 2; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        ctx->decode(token);
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::steady_clock::now();
        if (i >= 2) samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        token = static_cast<std::int64_t>(1000 + (i % 4096));
    }
    row.decode_ms = median_ms(samples);
    row.tok_s = 1000.0 / row.decode_ms;
    if (oscar) row.split_class = ctx->oscar_telemetry().last_split_class;
    // Greedy spot-check (INFO only): long-context chaos dominates, never gated.
    const auto logits = ctx->logits_host();
    row.greedy = static_cast<std::int64_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    for (float v : logits) {
        if (!std::isfinite(v)) throw std::runtime_error("sweep non-finite logits");
    }
    return row;
}

}  // namespace

int main() {
    try {
        const auto target = env("NINFER_EXL3_TARGET_PATH");
        if (target.empty()) {
            std::cerr << "sweep skipped: set NINFER_EXL3_TARGET_PATH\n";
            return 77;
        }
        const char* only = std::getenv("NINFER_E4C1_SWEEP_ONLY");
        const auto model = Exl3TextModel::load(target, 32768);
        std::vector<int> contexts = {512, 2048, 4096, 8192, 16384, 32768};
        if (only != nullptr && only[0] != 0) {
            contexts.clear();
            contexts.push_back(std::atoi(only));
        }
        const bool fresh = !std::filesystem::exists("results/oscar/E4C1_EXL3_OSCAR_CONTEXT_SWEEP.csv");
        std::ofstream csv("results/oscar/E4C1_EXL3_OSCAR_CONTEXT_SWEEP.csv", std::ios::app);
        require(static_cast<bool>(csv), "cannot write sweep CSV");
        if (fresh) {
            csv << "context,mode,prefill_ms,prefill_tok_s,decode_ms,tok_s,vram_bytes,split_class,greedy\n";
        }
        const char* modes = std::getenv("NINFER_E4C1_SWEEP_MODES");
        const std::string mode_filter = modes != nullptr ? modes : "both";
        for (int context : contexts) {
            const int decode_tokens = context >= 16384 ? 4 : 8;
            for (bool oscar : {false, true}) {
                if (mode_filter == "f16" && oscar) continue;
                if (mode_filter == "oscar" && !oscar) continue;
                const SweepRow row = run_point(*model, context, oscar, decode_tokens);
                csv << row.context << "," << (row.oscar ? "oscar" : "f16") << "," << row.prefill_ms
                    << "," << row.prefill_tok_s << "," << row.decode_ms << "," << row.tok_s << ","
                    << row.vram_bytes << "," << row.split_class << "," << row.greedy << "\n";
                csv.flush();
                std::cout << "E4C1 sweep context=" << row.context
                          << " greedy=" << row.greedy
                          << " mode=" << (row.oscar ? "oscar" : "f16")
                          << " prefill_ms=" << row.prefill_ms << " decode_ms=" << row.decode_ms
                          << " tok_s=" << row.tok_s << " vram_GB="
                          << static_cast<double>(row.vram_bytes) / 1e9
                          << " split=" << row.split_class << "\n";
            }
        }
        std::cout << "E4C1 context sweep: DONE\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "E4C1 context sweep: FAIL: " << error.what() << "\n";
        return 1;
    }
}
