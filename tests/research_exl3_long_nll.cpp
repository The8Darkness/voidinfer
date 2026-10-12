// Research tool: long-context teacher-forced NLL. Prefills the first
// LONG_NLL_CONTEXT tokens of LONG_NLL_IDS (16-row initial prefill, then
// 1024-row wide chunks), then teacher-forces LONG_NLL_CONTINUATION tokens one
// decode row at a time, recording -log p(token) and the argmax per row.
// Comparing runs (FP16 vs L0 OSCAR, or two L0 rotation sets) on the same ids
// gives paired long-context quality. LONG_NLL_ROWS=k (2..8) feeds k teacher
// rows per step through the multi-row verifier route instead and scores the
// token after each step. Not a test.
#include "exl3/engine_profile.h"
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
        voidinfer::exl3::apply_engine_profile();
        using namespace voidinfer::exl3;
        const char* target = std::getenv("VOIDINFER_EXL3_TARGET_PATH");
        const char* ids_path = std::getenv("LONG_NLL_IDS");
        const char* out_path = std::getenv("LONG_NLL_OUT");
        if (!target || !ids_path || !out_path) { std::cerr << "set VOIDINFER_EXL3_TARGET_PATH, LONG_NLL_IDS, LONG_NLL_OUT\n"; return 77; }
        const int context = std::atoi(std::getenv("LONG_NLL_CONTEXT") ? std::getenv("LONG_NLL_CONTEXT") : "32768");
        const int continuation = std::atoi(std::getenv("LONG_NLL_CONTINUATION") ? std::getenv("LONG_NLL_CONTINUATION") : "256");
        std::vector<std::int64_t> ids;
        std::ifstream f(ids_path);
        for (std::int64_t t; f >> t;) ids.push_back(t);
        if (static_cast<int>(ids.size()) < context + continuation) throw std::runtime_error("ids shorter than context + continuation");
        auto model = Exl3TextModel::load(target, context + continuation + 16);
        auto ctx = model->create_context(true);
        const std::span<const std::int64_t> all(ids.data(), ids.size());
        // LONG_NLL_LAYER_MAJOR=1 ingests the context like a served fresh prompt
        // (layer-major chunks with the prefill numerical policy, e.g.
        // VOIDINFER_EXL3_PREFILL_NVFP4); otherwise full-precision wide chunks.
        const bool layer_major = std::getenv("LONG_NLL_LAYER_MAJOR") != nullptr;
        if (layer_major) {
            const int initial = Exl3TextContext::layer_major_initial_rows();
            if (initial) ctx->prefill(all.first(initial));
            ctx->append_prefill_layer_major(all.subspan(initial, context - initial));
        } else {
            ctx->prefill(all.first(16));
            for (int at = 16; at < context;) {
                const int rows = std::min(1024, context - at);
                ctx->append_prefill_wide(all.subspan(at, rows));
                at += rows;
            }
        }
        cudaDeviceSynchronize();
        // LONG_NLL_ROW_SCORES_OUT: feed the continuation in LONG_NLL_ROWS-row
        // steps through the verifier route and write, per row, the argmax and
        // the logit gap of the scored token (the next teacher token, or the
        // matching entry of LONG_NLL_NEXT_IDS, e.g. another tier's argmaxes),
        // as VeriCache checks drafted rows (exact_row_scores).
        if (const char* rows_path = std::getenv("LONG_NLL_ROW_SCORES_OUT")) {
            const int step = std::getenv("LONG_NLL_ROWS") ? std::atoi(std::getenv("LONG_NLL_ROWS")) : 8;
            std::vector<std::int64_t> scored_ids(ids.begin() + context + 1, ids.begin() + context + continuation + 1);
            if (const char* next_path = std::getenv("LONG_NLL_NEXT_IDS")) {
                std::ifstream nf(next_path);
                scored_ids.clear();
                for (std::int64_t t; nf >> t;) scored_ids.push_back(t);
                if (static_cast<int>(scored_ids.size()) < continuation) throw std::runtime_error("LONG_NLL_NEXT_IDS too short");
            }
            ctx->prepare_continuation(step);
            std::ofstream rows_out(rows_path);
            for (int i = 0; i + step <= continuation; i += step) {
                ctx->append_prefill(all.subspan(static_cast<std::size_t>(context + i), static_cast<std::size_t>(step)));
                const auto scores = ctx->exact_row_scores(
                    std::span<const std::int64_t>(scored_ids.data() + i, static_cast<std::size_t>(step)), nullptr, 1.0f);
                for (int r = 0; r < step; ++r)
                    rows_out << scores[r].greedy << ' ' << scores[r].gap << ' ' << scored_ids[i + r] << '\n';
            }
            std::cout << "LONG_NLL_ROWS_DONE context=" << context << " rows=" << continuation << '\n';
            return 0;
        }
        std::ofstream out(out_path);
        // LONG_NLL_TOPK_OUT: per scored position the top-LONG_NLL_TOPK (default
        // 60) token ids (int32) then their log-probabilities (float32), for KL
        // against a reference record.
        const char* topk_path = std::getenv("LONG_NLL_TOPK_OUT");
        const int topk = std::getenv("LONG_NLL_TOPK") ? std::atoi(std::getenv("LONG_NLL_TOPK")) : 60;
        std::ofstream topk_out;
        if (topk_path) topk_out.open(topk_path, std::ios::binary);
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
            if (topk_out) {
                std::vector<std::int32_t> order(logits.size());
                for (std::size_t v = 0; v < order.size(); ++v) order[v] = static_cast<std::int32_t>(v);
                std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                    [&](std::int32_t a, std::int32_t b) { return logits[a] > logits[b]; });
                std::vector<float> lp(static_cast<std::size_t>(topk));
                for (int j = 0; j < topk; ++j)
                    lp[j] = static_cast<float>(static_cast<double>(logits[order[j]]) - mx - std::log(sum));
                topk_out.write(reinterpret_cast<const char*>(order.data()), topk * sizeof(std::int32_t));
                topk_out.write(reinterpret_cast<const char*>(lp.data()), topk * sizeof(float));
            }
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
