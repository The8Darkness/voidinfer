// E4C1 Stage 27: same-build short-context A/B (ordinary F16 vs OSCAR eager).
#include "exl3/text_model.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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

double median_ms(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

// Prefill 14 + 24 timed eager decodes. Returns median ms/token.
double bench_mode(Exl3TextModel& model, bool oscar) {
    auto context = model.create_context(false);
    if (oscar) require(context->try_enable_oscar_from_environment(), "bench OSCAR enable failed");
    const std::vector<std::int64_t> prompt = {248045, 846, 198, 33963, 799, 2716, 2029, 883,
                                              47503, 13, 248046, 198, 248045, 74455};
    context->prefill(prompt);
    std::int64_t token = 846;
    context->decode(token);
    std::vector<double> samples;
    for (int i = 0; i < 24; ++i) {
        const auto start = std::chrono::steady_clock::now();
        context->decode(token);
        cudaDeviceSynchronize();
        const auto end = std::chrono::steady_clock::now();
        if (i >= 4) {
            samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        token = static_cast<std::int64_t>(1000 + i);
    }
    return median_ms(samples);
}

}  // namespace

int main() {
    try {
        const auto target = env("NINFER_EXL3_TARGET_PATH");
        if (target.empty()) {
            std::cerr << "bench skipped: set NINFER_EXL3_TARGET_PATH\n";
            return 77;
        }
        const auto model = Exl3TextModel::load(target, 256);
        const double f16 = bench_mode(*model, false);
        const double oscar = bench_mode(*model, true);
        std::cout << "E4C1 short A/B: ordinary_f16_ms=" << f16 << " oscar_eager_ms=" << oscar
                  << " ratio=" << oscar / f16 << " f16_tok_s=" << 1000.0 / f16
                  << " oscar_tok_s=" << 1000.0 / oscar << "\n";
        std::cout << "E4C1 short A/B: DONE\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "E4C1 short A/B: FAIL: " << error.what() << "\n";
        return 1;
    }
}
