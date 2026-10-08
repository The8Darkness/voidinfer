// Research tool: can a VeriCache verifier pass overlap memory-bound decode?
// Two contexts of one model share the GPU: D decodes greedy tokens on a
// high-priority stream, V repeatedly runs an exact-history wide pass of
// OVL_ROWS rows (+ row scores) from a saved root on a low-priority stream.
// OVL_MODE 0 = D alone, 1 = V alone, 2 = both concurrently (V loops until D
// finishes). Prints D ms/token and V ms/pass. Not a test.
#include "exl3/text_model.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <mutex>
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
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
}  // namespace

int main() {
    try {
        using namespace ninfer::exl3;
        const char* target = std::getenv("NINFER_EXL3_TARGET_PATH");
        const char* ids_path = std::getenv("VERI_IDS");
        if (!target || !ids_path) { std::cerr << "set NINFER_EXL3_TARGET_PATH, VERI_IDS\n"; return 77; }
        const int context = env_int("OVL_CONTEXT", 16000);
        const int tokens = env_int("OVL_TOKENS", 256);
        const int rows = env_int("OVL_ROWS", 1024);
        const int mode = env_int("OVL_MODE", 2);
        const int v_passes = env_int("OVL_PASSES", 4);
        std::vector<std::int64_t> ids;
        std::ifstream f(ids_path);
        for (std::int64_t t; f >> t;) ids.push_back(t);
        if (static_cast<int>(ids.size()) < context + rows) throw std::runtime_error("ids too short");
        auto model = Exl3TextModel::load(target, context + std::max(tokens, rows) + 64);
        int lo = 0, hi = 0;
        check(cudaDeviceGetStreamPriorityRange(&lo, &hi), "priority range");
        cudaStream_t sd = nullptr, sv = nullptr;
        // OVL_V_SMS > 0: V and D run in disjoint green-context SM partitions.
        const int v_sms = env_int("OVL_V_SMS", 0);
        if (v_sms > 0) {
            check(cudaFree(nullptr), "init");
            CUdevice dev = 0;
            if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) throw std::runtime_error("cuDeviceGet");
            CUdevResource all{}, group{}, rest{};
            if (cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM) != CUDA_SUCCESS) throw std::runtime_error("get sm resource");
            unsigned groups = 1;
            if (cuDevSmResourceSplitByCount(&group, &groups, &all, &rest, 0, static_cast<unsigned>(v_sms)) != CUDA_SUCCESS)
                throw std::runtime_error("split sm");
            CUdevResourceDesc vd{}, dd{};
            CUgreenCtx vg{}, dg{};
            if (cuDevResourceGenerateDesc(&vd, &group, 1) != CUDA_SUCCESS || cuDevResourceGenerateDesc(&dd, &rest, 1) != CUDA_SUCCESS)
                throw std::runtime_error("resource desc");
            if (cuGreenCtxCreate(&vg, vd, dev, CU_GREEN_CTX_DEFAULT_STREAM) != CUDA_SUCCESS ||
                cuGreenCtxCreate(&dg, dd, dev, CU_GREEN_CTX_DEFAULT_STREAM) != CUDA_SUCCESS)
                throw std::runtime_error("green ctx");
            CUstream a{}, b{};
            if (cuGreenCtxStreamCreate(&a, dg, CU_STREAM_NON_BLOCKING, hi) != CUDA_SUCCESS ||
                cuGreenCtxStreamCreate(&b, vg, CU_STREAM_NON_BLOCKING, lo) != CUDA_SUCCESS)
                throw std::runtime_error("green stream");
            sv = b; (void)a;
            // D keeps the whole GPU (its cooperative grids are sized for every SM).
            check(cudaStreamCreateWithPriority(&sd, cudaStreamNonBlocking, hi), "stream d");
            std::cout << "OVL green v_sms=" << group.sm.smCount << " d_sms=" << rest.sm.smCount << '\n';
        } else {
            check(cudaStreamCreateWithPriority(&sd, cudaStreamNonBlocking, hi), "stream d");
            check(cudaStreamCreateWithPriority(&sv, cudaStreamNonBlocking, lo), "stream v");
        }
        const std::span<const std::int64_t> prompt(ids.data(), static_cast<std::size_t>(context));
        const auto ingest = [&](Exl3TextContext& ctx, cudaStream_t s, std::size_t shift = 0) {
            const std::span<const std::int64_t> prompt(ids.data() + shift, static_cast<std::size_t>(context));
            ctx.prefill(prompt.first(16), s);
            for (int at = 16; at < context;) {
                const int n = std::min(1024, context - at);
                ctx.append_prefill_wide(prompt.subspan(at, n), s);
                at += n;
            }
            check(cudaStreamSynchronize(s), "ingest");
        };
        const auto free_mib = [] { std::size_t f = 0, t = 0; cudaMemGetInfo(&f, &t); return static_cast<long long>(f >> 20); };
        std::cerr << "OVL free after load " << free_mib() << " MiB\n";
        std::unique_ptr<Exl3TextContext> d, v;
        if (mode != 1) { d = model->create_context(true); std::cerr << "OVL free after D create " << free_mib() << " MiB\n"; d->prepare_continuation(8); ingest(*d, sd); std::cerr << "OVL free after D ingest " << free_mib() << " MiB\n"; }
        if (mode != 0) {
            v = model->create_context(true); std::cerr << "OVL free after V create " << free_mib() << " MiB\n"; v->prepare_continuation(8); ingest(*v, sd, (mode >= 5 && !env_int("OVL_SAME_PROMPT", 0)) ? 4096 : 0); std::cerr << "OVL free after V ingest " << free_mib() << " MiB\n";
            v->save_verified_root(sv);
            check(cudaStreamSynchronize(sv), "root");
        }
        const std::span<const std::int64_t> block(ids.data() + context, static_cast<std::size_t>(rows));
        std::vector<std::int64_t> next(block.begin() + 1, block.end());
        next.push_back(-1);
        std::atomic<bool> stop{false};
        std::vector<double> pass_ms;
        const auto verifier = [&] {
            for (int pass = 0; mode == 2 ? !stop.load() : pass < v_passes; ++pass) {
                const auto start = Clock::now();
                v->restore_verified_root(sv);
                v->set_l0_exact_history(true);
                v->append_prefill_wide(block, sv);
                (void)v->exact_row_scores(next, sv);
                v->set_l0_exact_history(false);
                check(cudaStreamSynchronize(sv), "pass");
                pass_ms.push_back(ms(start));
            }
        };
        double d_ms = 0;
        const auto decoder = [&] {
            const auto start = Clock::now();
            for (int i = 0; i < tokens; ++i) {
                const auto logits = d->logits_host(sd);
                const auto token = static_cast<std::int64_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
                d->decode(token, sd);
            }
            check(cudaStreamSynchronize(sd), "decode");
            d_ms = ms(start);
        };
        if (mode == 6 || mode == 7) {
            // Concurrency correctness: two contexts, each decoding (mode 6: M1
            // decode; mode 7: 8-row continuations) alone and then concurrently
            // on two threads from the same roots; the token streams must match.
            const int steps = env_int("OVL_STEPS", 32);
            const auto first_token = [&](Exl3TextContext& ctx, cudaStream_t s) {
                const auto logits = ctx.logits_host(s);
                return static_cast<std::int64_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            };
            const std::int64_t first_a = first_token(*d, sd), first_b = first_token(*v, sv);
            d->save_verified_root(sd); v->save_verified_root(sv);
            check(cudaDeviceSynchronize(), "roots");
            std::mutex step_lock; const bool lock_steps = env_int("OVL_LOCK", 0) != 0;
            const auto run = [&](Exl3TextContext& ctx, cudaStream_t s, std::vector<std::int64_t>& out) {
                for (int i = 0; i < steps; ++i) {
                    std::unique_lock guard(step_lock, std::defer_lock); if (lock_steps) guard.lock();
                    std::int64_t token = &ctx == d.get() ? first_a : first_b;
                    if (i > 0) {
                        const auto logits = ctx.logits_host(s);
                        token = static_cast<std::int64_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
                    }
                    out.push_back(token);
                    if (mode == 6) ctx.decode(token, s);
                    else {
                        std::vector<std::int64_t> rows(8, token);
                        for (int r = 1; r < 8; ++r) rows[r] = ids[static_cast<std::size_t>(context + i * 8 + r)];
                        ctx.continue_rows(rows, s);
                    }
                    if (lock_steps) check(cudaStreamSynchronize(s), "locked step");
                }
                check(cudaStreamSynchronize(s), "run");
            };
            std::vector<std::int64_t> sa, sb, ca, cb;
            run(*d, sd, sa); run(*v, sv, sb);
            d->restore_verified_root(sd); v->restore_verified_root(sv);
            check(cudaDeviceSynchronize(), "rewind");
            if (env_int("OVL_PREFILL_LOAD", 0)) {
                // B ingests more prompt (wide prefill chunks) while A decodes.
                std::atomic<bool> done{false};
                std::thread loader([&] {
                    int at = context + 4096;
                    while (!done.load()) {
                        const std::span<const std::int64_t> chunk(ids.data() + at, 1024);
                        v->append_prefill_wide(chunk, sv);
                        check(cudaStreamSynchronize(sv), "load sync");
                        at += 1024;
                        if (v->position() + 1024 >= v->max_context()) break;
                    }
                });
                run(*d, sd, ca);
                done = true;
                loader.join();
                cb = sb;
            } else            if (env_int("OVL_BURN", 0)) {
                // Contention only: B floods the GPU with unrelated memsets.
                std::atomic<bool> done{false};
                void* junk = nullptr;
                check(cudaMalloc(&junk, 512u << 20), "burn buffer");
                std::thread burner([&] {
                    while (!done.load()) {
                        for (int i = 0; i < 8; ++i) check(cudaMemsetAsync(junk, i, 512u << 20, sv), "burn");
                        check(cudaStreamSynchronize(sv), "burn sync");
                    }
                });
                run(*d, sd, ca);
                done = true;
                burner.join();
                cudaFree(junk);
                cb = sb;
            } else if (env_int("OVL_SERIAL", 0)) { run(*d, sd, ca); run(*v, sv, cb); }
            else {
                std::thread other([&] { run(*v, sv, cb); });
                run(*d, sd, ca);
                other.join();
            }
            std::cout << "OVL cross-context same_stream=" << (sa == sb) << " ";
            std::cout << "OVL concurrency mode=" << mode << " steps=" << steps << " A_equal=" << (sa == ca)
                      << " B_equal=" << (sb == cb) << '\n';
            return 0;
        }
        if (mode == 5) {
            // Batched multi-agent continuation vs. two separate continuations
            // from the same verified roots (logits per row, then timing).
            const int n = env_int("OVL_BATCH_ROWS", 8);
            const std::span<const std::int64_t> ta(ids.data() + context, static_cast<std::size_t>(n));
            const std::span<const std::int64_t> tb(ids.data() + context + 64, static_cast<std::size_t>(n));
            if (env_int("OVL_GRAPH", 1)) d->prepare_batched_continuation_graph(*v, n, n);
            d->save_verified_root(sd); v->save_verified_root(sd);
            check(cudaStreamSynchronize(sd), "roots");
            const auto separate = [&] { d->continue_rows(ta, sd); v->continue_rows(tb, sd); };
            const auto batched = [&] { d->continue_rows_batched(*v, ta, tb, sd); };
            const auto rewind = [&] { d->restore_verified_root(sd); v->restore_verified_root(sd); };
            separate();
            const auto ra = d->continuation_logits_host(sd), rb = v->continuation_logits_host(sd);
            rewind();
            batched();
            const auto ba = d->continuation_logits_host(sd), bb = v->continuation_logits_host(sd);
            rewind();
            const auto compare = [&](const std::vector<float>& x, const std::vector<float>& y, const char* who) {
                const std::size_t vocab = x.size() / static_cast<std::size_t>(n);
                double max_abs = 0, sum_abs = 0; int agree = 0;
                for (int r = 0; r < n; ++r) {
                    const auto* xr = x.data() + r * vocab; const auto* yr = y.data() + r * vocab;
                    agree += std::max_element(xr, xr + vocab) - xr == std::max_element(yr, yr + vocab) - yr;
                    for (std::size_t j = 0; j < vocab; ++j) { const double e = std::abs(double(xr[j]) - yr[j]); max_abs = std::max(max_abs, e); sum_abs += e; }
                }
                std::cout << "OVL batch " << who << " rows=" << n << " argmax_agree=" << agree << "/" << n
                          << " max_abs_logit_diff=" << max_abs << " mean_abs=" << sum_abs / double(x.size()) << " sizes=" << x.size() << "/" << y.size() << '\n';
            };
            compare(ra, ba, "own");
            compare(rb, bb, "peer");
            const auto time = [&](auto&& f, const char* what) {
                for (int i = 0; i < 3; ++i) { f(); rewind(); }
                check(cudaStreamSynchronize(sd), "warm");
                const auto start = Clock::now();
                for (int i = 0; i < 20; ++i) { f(); rewind(); }
                check(cudaStreamSynchronize(sd), "timed");
                std::cout << "OVL batch time " << what << " ms=" << ms(start) / 20 << " (incl. rewind)\n";
            };
            if (const char* only = std::getenv("OVL_ONLY")) {
                const bool b = std::string(only) == "batched";
                for (int i = 0; i < 10; ++i) { if (b) batched(); else separate(); rewind(); }
                check(cudaStreamSynchronize(sd), "profile loop");
                return 0;
            }
            time(separate, "separate");
            time(batched, "batched");
            time(rewind, "rewind-only");
            return 0;
        }
        if (mode == 4) {
            // Cost of one target continuation forward by row count (batched-round estimate).
            for (const int n : {1, 2, 4, 8, 9, 12, 16, 32}) {
                if (n == 1) {
                    const auto start = Clock::now();
                    for (int i = 0; i < 20; ++i) d->decode(ids[static_cast<std::size_t>(context + i)], sd);
                    check(cudaStreamSynchronize(sd), "m1");
                    std::cout << "OVL rows=1 ms=" << ms(start) / 20 << '\n';
                    continue;
                }
                const std::span<const std::int64_t> chunk(ids.data() + context, static_cast<std::size_t>(n));
                const auto run = [&] { if (n <= 8) d->continue_rows(chunk, sd); else d->append_prefill_wide(chunk, sd); };
                run(); check(cudaStreamSynchronize(sd), "warm");
                const auto start = Clock::now();
                for (int i = 0; i < 20; ++i) run();
                check(cudaStreamSynchronize(sd), "rows");
                std::cout << "OVL rows=" << n << " ms=" << ms(start) / 20 << '\n';
            }
            return 0;
        }
        if (mode == 3) {
            // Both contexts decode greedy tokens concurrently (time-sliced lanes).
            double v_ms = 0;
            std::thread other([&] {
                const auto start = Clock::now();
                for (int i = 0; i < tokens; ++i) {
                    const auto logits = v->logits_host(sv);
                    v->decode(static_cast<std::int64_t>(std::max_element(logits.begin(), logits.end()) - logits.begin()), sv);
                }
                check(cudaStreamSynchronize(sv), "decode v");
                v_ms = ms(start);
            });
            decoder();
            other.join();
            std::cout << "OVL mode=3 d_ms_per_token=" << d_ms / tokens << " v_ms_per_token=" << v_ms / tokens
                      << " aggregate_tok_s=" << 2.0 * tokens * 1000.0 / std::max(d_ms, v_ms) << '\n';
            return 0;
        }
        if (mode == 0) decoder();
        else if (mode == 1) verifier();
        else {
            std::thread vt(verifier);
            decoder();
            stop = true;
            vt.join();
        }
        std::cout << "OVL mode=" << mode << " context=" << context << " rows=" << rows;
        if (mode != 1) std::cout << " d_ms_per_token=" << d_ms / tokens;
        if (!pass_ms.empty()) {
            double sum = 0;
            for (const auto p : pass_ms) sum += p;
            std::cout << " v_passes=" << pass_ms.size() << " v_ms_per_pass=" << sum / pass_ms.size()
                      << " v_first=" << pass_ms.front();
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "OVL FAIL: " << e.what() << '\n';
        return 1;
    }
}
