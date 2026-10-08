// Research tool: VeriCache block verification on an L0 OSCAR context.
// Prefills VERI_CONTEXT tokens of VERI_IDS, then generates VERI_TOKENS greedy
// tokens: VERI_BLOCK tokens at a time with L0 attention, after which the block
// is re-run from the last verified root with exact FP16-L2 history attention
// (one wide pass) and checked row by row. Mode exact: the first token that
// differs from the exact greedy token is replaced and the rest discarded.
// Mode tolerance (VERI_DELTA > 0): only tokens whose exact logit trails the
// exact top logit by more than VERI_DELTA are replaced. VERI_EXACT_PREFILL=1
// ingests the prompt with exact history too. Writes the generated tokens to
// VERI_OUT and prints per-stage timings. Not a test.
#include "exl3/text_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return v ? std::atoi(v) : fallback;
}
std::int64_t argmax(const std::vector<float>& logits) {
    return static_cast<std::int64_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}
}  // namespace

int main() {
    try {
        using namespace ninfer::exl3;
        const char* target = std::getenv("NINFER_EXL3_TARGET_PATH");
        const char* ids_path = std::getenv("VERI_IDS");
        const char* out_path = std::getenv("VERI_OUT");
        if (!target || !ids_path || !out_path) { std::cerr << "set NINFER_EXL3_TARGET_PATH, VERI_IDS, VERI_OUT\n"; return 77; }
        const int context = env_int("VERI_CONTEXT", 16000);
        const int tokens = env_int("VERI_TOKENS", 512);
        const int block = env_int("VERI_BLOCK", 32);
        const float delta = std::getenv("VERI_DELTA") ? std::strtof(std::getenv("VERI_DELTA"), nullptr) : 0.0f;
        const bool verify = env_int("VERI_VERIFY", 1) != 0;
        const bool exact_prefill = env_int("VERI_EXACT_PREFILL", 0) != 0;
        std::vector<std::int64_t> ids;
        std::ifstream f(ids_path);
        for (std::int64_t t; f >> t;) ids.push_back(t);
        if (static_cast<int>(ids.size()) < context) throw std::runtime_error("ids shorter than context");
        auto model = Exl3TextModel::load(target, context + tokens + block + 64);
        auto ctx = model->create_context(true);
        ctx->prepare_continuation(8);
        const std::span<const std::int64_t> prompt(ids.data(), static_cast<std::size_t>(context));
        auto start = Clock::now();
        if (exact_prefill) ctx->set_l0_exact_history(true);
        ctx->prefill(prompt.first(16));
        for (int at = 16; at < context;) {
            const int rows = std::min(1024, context - at);
            ctx->append_prefill_wide(prompt.subspan(at, rows));
            at += rows;
        }
        ctx->set_l0_exact_history(false);
        cudaDeviceSynchronize();
        const double prefill_ms = ms(start);

        std::vector<std::int64_t> out;
        double gen_ms = 0, export_ms = 0, restore_ms = 0, verify_ms = 0, fix_ms = 0;
        int blocks = 0, corrections = 0, discarded = 0;
        std::vector<float> gaps;
        start = Clock::now();
        // Verified root: device GDN checkpoint (VERI_HOST_ROOT=1: host export).
        const bool host_root = env_int("VERI_HOST_ROOT", 0) != 0;
        std::shared_ptr<const Exl3ExactHostState> root;
        std::int64_t root_greedy = argmax(ctx->logits_host());
        const auto save_root = [&] {
            if (host_root) root = ctx->export_exact_host_state();
            else ctx->save_verified_root();
            root_greedy = argmax(ctx->logits_host());
        };
        const auto restore_root = [&] {
            if (host_root) ctx->restore_exact_host_state(*root);
            else ctx->restore_verified_root();
        };
        if (verify) save_root();
        cudaDeviceSynchronize();
        export_ms += ms(start);
        while (static_cast<int>(out.size()) < tokens) {
            // Draft: block tokens with L0 attention.
            std::vector<std::int64_t> draft;
            start = Clock::now();
            const int count = std::min(block, tokens - static_cast<int>(out.size()));
            for (int i = 0; i < count; ++i) {
                const auto token = argmax(ctx->logits_host());
                draft.push_back(token);
                ctx->decode(token);
            }
            cudaDeviceSynchronize();
            gen_ms += ms(start);
            if (!verify) { out.insert(out.end(), draft.begin(), draft.end()); continue; }
            ++blocks;
            // Verify: exact pass over the block from the verified root.
            start = Clock::now();
            restore_root();
            cudaDeviceSynchronize();
            restore_ms += ms(start);
            start = Clock::now();
            ctx->set_l0_exact_history(true);
            const std::span<const std::int64_t> rows(draft);
            if (rows.size() > 8) ctx->append_prefill_wide(rows);
            else ctx->append_prefill(rows);
            std::vector<std::int64_t> next(draft.begin() + 1, draft.end());
            next.push_back(-1);
            const auto scores = ctx->exact_row_scores(next);
            ctx->set_l0_exact_history(false);
            verify_ms += ms(start);
            // First rejected position: row i predicts draft[i + 1].
            int reject = -1;
            std::int64_t correction = -1;
            if (draft[0] != root_greedy) { reject = 0; correction = root_greedy; }
            else for (int i = 0; i + 1 < count; ++i) {
                gaps.push_back(scores[i].gap);
                const bool bad = delta > 0 ? scores[i].gap > delta : scores[i].greedy != draft[i + 1];
                if (bad) { reject = i + 1; correction = scores[i].greedy; break; }
            }
            if (reject < 0) {
                out.insert(out.end(), draft.begin(), draft.end());
            } else {
                ++corrections;
                discarded += count - reject;
                start = Clock::now();
                std::vector<std::int64_t> fixed(draft.begin(), draft.begin() + reject);
                fixed.push_back(correction);
                restore_root();
                ctx->set_l0_exact_history(true);
                if (fixed.size() > 8) ctx->append_prefill_wide(fixed);
                else ctx->append_prefill(fixed);
                ctx->set_l0_exact_history(false);
                cudaDeviceSynchronize();
                fix_ms += ms(start);
                out.insert(out.end(), fixed.begin(), fixed.end());
            }
            start = Clock::now();
            save_root();
            cudaDeviceSynchronize();
            export_ms += ms(start);
        }
        out.resize(static_cast<std::size_t>(tokens));
        std::ofstream o(out_path);
        for (const auto t : out) o << t << '\n';
        std::sort(gaps.begin(), gaps.end());
        const auto pct = [&](double q) { return gaps.empty() ? 0.0 : gaps[static_cast<std::size_t>(q * (gaps.size() - 1))]; };
        std::cout << "VERI context=" << context << " tokens=" << tokens << " block=" << block << " delta=" << delta
                  << " exact_prefill=" << exact_prefill << " verify=" << verify << "\n"
                  << "VERI prefill_ms=" << prefill_ms << " gen_ms=" << gen_ms << " (" << gen_ms / tokens << " ms/token)"
                  << " restore_ms=" << restore_ms << " verify_ms=" << verify_ms << " fix_ms=" << fix_ms
                  << " export_ms=" << export_ms << "\n"
                  << "VERI blocks=" << blocks << " corrections=" << corrections << " discarded=" << discarded
                  << " nonzero_gaps=" << std::count_if(gaps.begin(), gaps.end(), [](float g) { return g > 0; })
                  << " gap_p99=" << pct(0.99) << " gap_max=" << (gaps.empty() ? 0.0f : gaps.back()) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "VERI FAIL: " << e.what() << '\n';
        return 1;
    }
}
