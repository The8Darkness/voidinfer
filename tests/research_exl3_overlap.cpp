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
        if (mode == 8) {
            // N-agent batched continuation (OVL_AGENTS contexts, owner + peers)
            // vs. separate continuations from the same verified roots.
            const int agents = env_int("OVL_AGENTS", 4);
            const int n = env_int("OVL_BATCH_ROWS", 8);
            std::vector<std::unique_ptr<Exl3TextContext>> extra;
            std::vector<Exl3TextContext*> all{d.get(), v.get()};
            for (int a = 2; a < agents; ++a) {
                extra.push_back(model->create_context(true));
                extra.back()->prepare_continuation(8);
                ingest(*extra.back(), sd, static_cast<std::size_t>(a) * 2048);
                all.push_back(extra.back().get());
            }
            std::cerr << "OVL free after " << agents << " contexts " << free_mib() << " MiB\n";
            std::vector<std::span<const std::int64_t>> rows_of;
            for (int a = 0; a < agents; ++a)
                rows_of.emplace_back(ids.data() + context + 64 * a, static_cast<std::size_t>(n));
            std::vector<Exl3TextContext*> peer_list(all.begin() + 1, all.end());
            std::vector<int> peer_rows(peer_list.size(), n);
            if (env_int("OVL_GRAPH", 1)) d->prepare_batched_continuation_graph(peer_list, n, peer_rows);
            for (auto* c : all) c->save_verified_root(sd);
            check(cudaStreamSynchronize(sd), "roots");
            std::vector<Exl3TextContext::BatchedPeer> peers;
            for (int a = 1; a < agents; ++a) peers.push_back({all[a], rows_of[a], nullptr});
            // OVL_SIBLINGS=1: each agent's last two rows are sibling leaves
            // (verification tree layout) as in the Engine's DFlash2 rounds.
            const bool siblings = env_int("OVL_SIBLINGS", 0) != 0;
            // OVL_SIB_A / OVL_SIB_B: the two sibling leaves' chain offsets.
            const std::vector<int> sibling_offsets{env_int("OVL_SIB_A", 1), env_int("OVL_SIB_B", 3)};
            const auto arm = [&] { if (siblings) for (auto* c : all) c->set_verifier_siblings(sibling_offsets); };
            const auto separate = [&] { arm(); for (int a = 0; a < agents; ++a) all[a]->continue_rows(rows_of[a], sd); };
            const auto batched = [&] { arm(); d->continue_rows_batched(peers, rows_of[0], sd); };
            const auto rewind = [&] { for (auto* c : all) c->restore_verified_root(sd); };
            separate();
            std::vector<std::vector<float>> ref;
            for (auto* c : all) ref.push_back(c->continuation_logits_host(sd));
            rewind();
            batched();
            for (int a = 0; a < agents; ++a) {
                const auto got = all[a]->continuation_logits_host(sd);
                const std::size_t vocab = got.size() / static_cast<std::size_t>(n);
                double max_abs = 0; int agree = 0;
                for (int r = 0; r < n; ++r) {
                    const auto* x = ref[a].data() + r * vocab; const auto* y = got.data() + r * vocab;
                    agree += std::max_element(x, x + vocab) - x == std::max_element(y, y + vocab) - y;
                    for (std::size_t j = 0; j < vocab; ++j) max_abs = std::max(max_abs, std::abs(double(x[j]) - y[j]));
                }
                std::cout << "OVL nbatch agent=" << a << " argmax_agree=" << agree << "/" << n
                          << " max_abs_logit_diff=" << max_abs << '\n';
            }
            if (siblings) {
                // Ground truth of each sibling leaf: the plain continuation of
                // its path (chain rows 0..o-1, then the sibling token).
                std::vector<std::vector<float>> got;
                for (auto* c : all) got.push_back(c->continuation_logits_host(sd));
                const int chain = n - 2;
                for (int a = 0; a < agents; ++a) {
                    const std::size_t vocab = got[a].size() / static_cast<std::size_t>(n);
                    for (int j = 0; j < 2; ++j) {
                        const int o = sibling_offsets[static_cast<std::size_t>(j)];
                        std::vector<std::int64_t> path(rows_of[a].begin(), rows_of[a].begin() + o);
                        path.push_back(rows_of[a][static_cast<std::size_t>(chain + j)]);
                        rewind();
                        all[a]->continue_rows(path, sd);
                        const auto truth = all[a]->continuation_logits_host(sd);
                        const auto* t = truth.data() + (path.size() - 1) * vocab;
                        const auto score = [&](const float* y) {
                            const auto ty = std::max_element(t, t + vocab) - t, yy = std::max_element(y, y + vocab) - y;
                            return double(t[ty]) - t[yy];
                        };
                        const auto* bs = got[a].data() + static_cast<std::size_t>(chain + j) * vocab;
                        const auto* ss = ref[a].data() + static_cast<std::size_t>(chain + j) * vocab;
                        std::cout << "OVL sibling-truth agent=" << a << " sibling=" << j << " offset=" << o
                                  << " gap_batched=" << score(bs) << " gap_separate=" << score(ss) << '\n';
                    }
                }
            }
            rewind();
            const auto time = [&](auto&& f, const char* what) {
                for (int i = 0; i < 3; ++i) { f(); rewind(); }
                check(cudaStreamSynchronize(sd), "warm");
                const auto start = Clock::now();
                for (int i = 0; i < 20; ++i) { f(); rewind(); }
                check(cudaStreamSynchronize(sd), "timed");
                std::cout << "OVL nbatch time agents=" << agents << " " << what << " ms=" << ms(start) / 20 << " (incl. rewind)\n";
            };
            time(separate, "separate");
            time(batched, "batched");
            time(rewind, "rewind-only");
            return 0;
        }
        if (mode == 9) {
            // Multi-round transactions with partial retention (the Engine's
            // DFlash2 round shape): OVL_ROUNDS rounds of 8 rows per agent,
            // agent a keeps a deterministic 1..8 rows each round. Runs every
            // agent separately, rewinds, runs batched rounds, and compares
            // each round's logits (state divergence across rounds shows up).
            const int agents = env_int("OVL_AGENTS", 4);
            const int rounds = env_int("OVL_ROUNDS", 48);
            constexpr int n = 8;
            std::vector<std::unique_ptr<Exl3TextContext>> extra;
            std::vector<Exl3TextContext*> all{d.get(), v.get()};
            for (int a = 2; a < agents; ++a) {
                extra.push_back(model->create_context(true));
                extra.back()->prepare_continuation(8);
                ingest(*extra.back(), sd, static_cast<std::size_t>(a) * 2048);
                all.push_back(extra.back().get());
            }
            std::vector<Exl3TextContext*> peer_list(all.begin() + 1, all.end());
            if (env_int("OVL_GRAPH", 1)) d->prepare_batched_continuation_graph(peer_list, n, std::vector<int>(peer_list.size(), n));
            for (std::size_t a = 0; a < all.size(); ++a) {
                auto* c = all[a];
                c->prepare_transaction();
                c->set_gdn_repair_overlap(env_int("OVL_REPAIR_OVERLAP", 0) != 0);
                // A completed small forward (consistent tap/embedding rows)
                // precedes the first transaction, as in the Engine's rounds.
                c->continue_rows(std::span<const std::int64_t>(ids.data() + context + 4000 + 8 * a, 8), sd);
                std::cerr << "OVL warm agent=" << a << " position=" << c->position()
                          << " tap_rows=" << c->captured_tap_rows() << "\n";
            }
            for (auto* c : all) c->save_verified_root(sd);
            check(cudaStreamSynchronize(sd), "roots");
            const auto tokens_of = [&](int round, int a) {
                return std::span<const std::int64_t>(ids.data() + context + 512 * a + 8 * round,
                    static_cast<std::size_t>(n));
            };
            // OVL_SIBLINGS=1: the last two rows are sibling leaves at offsets
            // varying per round/agent; a retained depth equal to a sibling's
            // offset promotes that sibling first (the Engine's tree rounds).
            const bool tree = env_int("OVL_SIBLINGS", 0) != 0;
            const int chain_rows = tree ? n - 2 : n;
            const auto offsets_of = [&](int round, int a) {
                return std::vector<int>{1 + (round + a) % 3, 1 + (round * 3 + a) % 4};
            };
            const auto keep_of = [&](int round, int a) {
                const int keep = 1 + (round * 5 + a * 3) % n;
                return tree ? std::min(keep, chain_rows) : keep;
            };
            // OVL_STREAMS=1: every agent owns a stream (Engine lanes); the
            // batched forward runs on agent 0's stream after every member's
            // queued work and members resume after it (coordinator events).
            const bool own_streams = env_int("OVL_STREAMS", 1) != 0;
            std::vector<cudaStream_t> lane_streams(all.size(), sd);
            std::vector<cudaEvent_t> ready(all.size()), done(all.size());
            for (std::size_t a = 0; a < all.size(); ++a) {
                if (own_streams && a) check(cudaStreamCreateWithFlags(&lane_streams[a], cudaStreamNonBlocking), "lane stream");
                check(cudaEventCreateWithFlags(&ready[a], cudaEventDisableTiming), "ready");
                check(cudaEventCreateWithFlags(&done[a], cudaEventDisableTiming), "done");
            }
            // Device time of each round's target forward(s) (events on agent
            // 0's stream around the batched call; around each separate call).
            cudaEvent_t t0, t1;
            check(cudaEventCreate(&t0), "t0"); check(cudaEventCreate(&t1), "t1");
            double forward_ms[2] = {0, 0};
            const auto run = [&](bool batched) {
                std::vector<std::vector<std::vector<float>>> logits(rounds);
                for (int r = 0; r < rounds; ++r) {
                    for (std::size_t a = 0; a < all.size(); ++a) {
                        try { all[a]->begin_transaction(lane_streams[a]); }
                        catch (...) {
                            std::cerr << "OVL begin failed batched=" << batched << " round=" << r << " agent=" << a
                                      << " tap_rows=" << all[a]->captured_tap_rows() << std::endl;
                            throw;
                        }
                    }
                    if (tree) for (int a = 0; a < agents; ++a) all[a]->set_verifier_siblings(offsets_of(r, a));
                    if (batched) {
                        std::vector<Exl3TextContext::BatchedPeer> peers;
                        for (int a = 1; a < agents; ++a) {
                            check(cudaEventRecord(ready[a], lane_streams[a]), "member ready");
                            check(cudaStreamWaitEvent(lane_streams[0], ready[a], 0), "leader waits member");
                            peers.push_back({all[a], tokens_of(r, a), lane_streams[a]});
                        }
                        check(cudaEventRecord(t0, lane_streams[0]), "t0");
                        d->continue_rows_batched(peers, tokens_of(r, 0), lane_streams[0]);
                        check(cudaEventRecord(t1, lane_streams[0]), "t1");
                        check(cudaEventSynchronize(t1), "t1 sync");
                        float elapsed = 0; check(cudaEventElapsedTime(&elapsed, t0, t1), "elapsed");
                        if (r >= 2) forward_ms[1] += elapsed;
                        check(cudaEventRecord(done[0], lane_streams[0]), "leader done");
                        for (int a = 1; a < agents; ++a)
                            check(cudaStreamWaitEvent(lane_streams[a], done[0], 0), "member resumes");
                    } else for (int a = 0; a < agents; ++a) {
                        check(cudaStreamSynchronize(lane_streams[a]), "separate pre");
                        check(cudaEventRecord(t0, lane_streams[a]), "t0");
                        all[a]->continue_rows(tokens_of(r, a), lane_streams[a]);
                        check(cudaEventRecord(t1, lane_streams[a]), "t1");
                        check(cudaEventSynchronize(t1), "t1 sync");
                        float elapsed = 0; check(cudaEventElapsedTime(&elapsed, t0, t1), "elapsed");
                        if (r >= 2) forward_ms[0] += elapsed;
                    }
                    for (int a = 0; a < agents; ++a) {
                        logits[r].push_back(all[a]->continuation_logits_host(lane_streams[a]));
                        const int keep = keep_of(r, a);
                        if (tree && keep >= 2) {
                            const auto offsets = offsets_of(r, a);
                            for (int j = 0; j < 2; ++j)
                                if (offsets[static_cast<std::size_t>(j)] == keep - 1 && (r + a) % 2 == 0) {
                                    all[a]->promote_sibling_row(chain_rows + j, keep - 1, lane_streams[a]);
                                    break;
                                }
                        }
                        if (keep < n) all[a]->retain_transaction_prefix(keep, lane_streams[a]);
                        all[a]->commit_transaction();
                    }
                }
                check(cudaDeviceSynchronize(), "rounds");
                return logits;
            };
            const auto reference = run(false);
            for (std::size_t a = 0; a < all.size(); ++a) all[a]->restore_verified_root(lane_streams[a]);
            check(cudaDeviceSynchronize(), "rewind");
            const auto got = run(true);
            int first_bad_round = -1;
            for (int r = 0; r < rounds; ++r)
                for (int a = 0; a < agents; ++a) {
                    const auto& x = reference[r][a]; const auto& y = got[r][a];
                    const std::size_t vocab = x.size() / n;
                    double max_abs = 0, max_gap = 0; int agree = 0;
                    for (int row = 0; row < n; ++row) {
                        const auto* xr = x.data() + row * vocab; const auto* yr = y.data() + row * vocab;
                        const auto rx = std::max_element(xr, xr + vocab) - xr, ry = std::max_element(yr, yr + vocab) - yr;
                        agree += rx == ry;
                        // VeriCache criterion: how far the batched choice trails
                        // the reference top logit.
                        max_gap = std::max(max_gap, double(xr[rx]) - xr[ry]);
                        for (std::size_t j = 0; j < vocab; ++j) max_abs = std::max(max_abs, std::abs(double(xr[j]) - yr[j]));
                    }
                    if (max_gap > 1.0) {
                        if (first_bad_round < 0) first_bad_round = r;
                        std::cout << "OVL rounds round=" << r << " agent=" << a << " keep=" << keep_of(r, a)
                                  << " argmax_agree=" << agree << "/" << n << " max_gap=" << max_gap
                                  << " max_abs_logit_diff=" << max_abs << '\n';
                    }
                }
            std::cout << "OVL rounds agents=" << agents << " rounds=" << rounds << " first_bad_round=" << first_bad_round << '\n';
            std::cout << "OVL forward ms/round separate=" << forward_ms[0] / std::max(1, rounds - 2)
                      << " batched=" << forward_ms[1] / std::max(1, rounds - 2) << '\n';
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
