// E4C1 Stage 12: profiler driver - ingest to C, then cudaProfilerApi-scoped decodes.
#include "exl3/text_model.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_profiler_api.h>
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

int env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != 0 ? std::atoi(value) : fallback;
}

}  // namespace

int main() {
    try {
        const auto target = env("NINFER_EXL3_TARGET_PATH");
        if (target.empty()) {
            std::cerr << "profile skipped: set NINFER_EXL3_TARGET_PATH\n";
            return 77;
        }
        const int context = env_int("NINFER_E4C1_PROF_CTX", 4096);
        const int decodes = env_int("NINFER_E4C1_PROF_DECODERS", 10);
        const auto model = Exl3TextModel::load(target, 32768);
        auto ctx = model->create_context(false);
        require(ctx->try_enable_oscar_from_environment(), "profile OSCAR enable failed");
        static const std::int64_t base[] = {248045, 846, 198, 33963, 799, 2716, 2029, 883,
                                            47503, 13, 248046, 198, 248045, 74455};
        std::uint64_t lcg = 0xC10CULL;
        // Chunked ingestion: 16-row prefill, then single decodes.
        {
            std::vector<std::int64_t> first16;
            for (int i = 0; i < 16; ++i) first16.push_back(base[i % 14]);
            ctx->prefill(first16);
        }
        for (int i = 16; i < context; ++i) {
            lcg = 1664525ULL * lcg + 1013904223ULL;
            ctx->decode(static_cast<std::int64_t>((lcg >> 16U) % 248320U));
        }
        std::cout << "E4C1 profile: ingested " << context << " split_class="
                  << ctx->oscar_telemetry().last_split_class << std::flush;
        if (cudaProfilerStart() != cudaSuccess) throw std::runtime_error("profiler start");
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < decodes; ++i) {
            ctx->decode(static_cast<std::int64_t>(2000 + i));
        }
        if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("profiler sync");
        const auto t1 = std::chrono::steady_clock::now();
        if (cudaProfilerStop() != cudaSuccess) throw std::runtime_error("profiler stop");
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / decodes;
        std::cout << " decodes=" << decodes << " ms_per_token=" << ms << " tok_s=" << 1000.0 / ms
                  << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "E4C1 profile: FAIL: " << error.what() << "\n";
        return 1;
    }
}
