// Research tool: long-context teacher-forced NLL. Prefills the first
// LONG_NLL_CONTEXT tokens of LONG_NLL_IDS (16-row initial prefill, then
// 1024-row wide chunks), then teacher-forces LONG_NLL_CONTINUATION tokens one
// decode row at a time, recording -log p(token) and the argmax per row.
// Comparing runs (FP16 vs L0 OSCAR, or two L0 rotation sets) on the same ids
// gives paired long-context quality. LONG_NLL_ROWS=k (2..8) feeds k teacher
// rows per step through the multi-row verifier route instead and scores the
// token after each step. Not a test.
#include "exl3/text_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    try {
        using namespace ninfer::exl3;
        const char* target = std::getenv("NINFER_EXL3_TARGET_PATH");
        const char* ids_path = std::getenv("LONG_NLL_IDS");
        const char* out_path = std::getenv("LONG_NLL_OUT");
        if (!target || !ids_path || !out_path) { std::cerr << "set NINFER_EXL3_TARGET_PATH, LONG_NLL_IDS, LONG_NLL_OUT\n"; return 77; }
        const int context = std::atoi(std::getenv("LONG_NLL_CONTEXT") ? std::getenv("LONG_NLL_CONTEXT") : "32768");
        const int continuation = std::atoi(std::getenv("LONG_NLL_CONTINUATION") ? std::getenv("LONG_NLL_CONTINUATION") : "256");
        std::vector<std::int64_t> ids;
        std::ifstream f(ids_path);
        for (std::int64_t t; f >> t;) ids.push_back(t);
        if (static_cast<int>(ids.size()) < context + continuation) throw std::runtime_error("ids shorter than context + continuation");
        auto model = Exl3TextModel::load(target, context + continuation + 16);
        auto ctx = model->create_context(true);
        const std::span<const std::int64_t> all(ids.data(), ids.size());
        ctx->prefill(all.first(16));
        for (int at = 16; at < context;) {
            const int rows = std::min(1024, context - at);
            ctx->append_prefill_wide(all.subspan(at, rows));
            at += rows;
        }
        cudaDeviceSynchronize();
        std::ofstream out(out_path);
        double total = 0.0;
        int scored = 0;
        for (int i = 0; i < continuation; ++i) {
            const auto logits = ctx->logits_host();
            const std::int64_t token = ids[context + i];
            float mx = -INFINITY;
            std::size_t arg = 0;
            for (std::size_t v = 0; v < logits.size(); ++v) if (logits[v] > mx) { mx = logits[v]; arg = v; }
            double sum = 0.0;
            for (const float l : logits) sum += std::exp(static_cast<double>(l) - mx);
            const double nll = -(static_cast<double>(logits[static_cast<std::size_t>(token)]) - mx - std::log(sum));
            total += nll;
            ++scored;
            // LONG_NLL_GREEDY=1: free greedy generation (token = argmax) and the
            // top-1/top-2 logit margin, for divergence studies between KV tiers.
            static const bool greedy = std::getenv("LONG_NLL_GREEDY") != nullptr;
            float second = -INFINITY;
            for (std::size_t v = 0; v < logits.size(); ++v) if (v != arg && logits[v] > second) second = logits[v];
            out << nll << ' ' << arg << ' ' << token << ' ' << (mx - second) << '\n';
            // LONG_NLL_ROWS=k (2..8): teacher-force k rows per verifier-route
            // step (append_prefill) and score only the token after each step.
            static const int step = std::getenv("LONG_NLL_ROWS") ? std::atoi(std::getenv("LONG_NLL_ROWS")) : 1;
            if (step > 1) {
                const int take = std::min(step, continuation - i);
                ctx->append_prefill(all.subspan(static_cast<std::size_t>(context + i), static_cast<std::size_t>(take)));
                i += take - 1;
                continue;
            }
            ctx->decode(greedy ? static_cast<std::int64_t>(arg) : token);
        }
        std::cout << "LONG_NLL context=" << context << " continuation=" << continuation
                  << " scored=" << scored << " mean_nll=" << total / scored << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "LONG_NLL FAIL: " << e.what() << '\n';
        return 1;
    }
}
