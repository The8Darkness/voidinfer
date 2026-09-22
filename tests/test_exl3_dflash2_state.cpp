 // E5A3: DFlash2 bounded draft-KV ring, logical rollback, and production-path
 // latency attribution on the qualified E5A2 baseline.
 //
 // Single draft instance at a time (VRAM headroom ~3.4 GiB forbids two loaded
 // drafts). Cross-run comparisons use saved FNV digests, never two models.
 //
 // Coverage:
 //   S1 commit+propose_cached == legacy propose (exact, small ring) +
 //      bulk-vs-incremental digest equality + propose-does-not-mutate proof
 //   S2 forced accept 0/1/partial(3)/full(7) vs clean replay (digest+proposal)
 //      + explicit rewind_to/recommit logical rollback
 //   S3 window rollover 2047/2048/2049/2060 + bulk prefill-skip + full wrap 4150
 //   S4 reset before-full/after-full/after-wrap + destroy/recreate
 //   S5 VRAM stages (target / +draft / bounded after wrap)
 //   S6 propose wall latency legacy-vs-ring + env-gated phase breakdown (LAST)
 #include "exl3/text_model.h"
 #include "exl3/dflash2_draft.h"
 
 #include <cuda_runtime.h>
 
 #include <algorithm>
 #include <array>
 #include <chrono>
 #include <cmath>
 #include <cstdint>
 #include <cstdlib>
 #include <cstring>
 #include <filesystem>
 #include <fstream>
 #include <iomanip>
 #include <iostream>
 #include <numeric>
 #include <sstream>
 #include <stdexcept>
 #include <string>
 #include <vector>
 
 namespace {
 
 using ninfer::exl3::Exl3TextModel;
 using ninfer::exl3::Exl3TextContext;
 using ninfer::exl3::Exl3Dflash2DraftModel;
 using ninfer::exl3::Exl3Dflash2TapHistory;
 
 constexpr int kHidden = 5120;
 constexpr int kVocab = 248320;
 constexpr int kTapCount = 5;
 constexpr int kMaskToken = 248070;
 constexpr int kBlockLen = 8;
 constexpr std::array<int, kTapCount> kTapLayers = {5, 19, 33, 47, 61};
 
 const std::vector<std::int64_t> kPrompt = {248045, 846, 198, 7734, 799, 11316, 883,
                                            12050, 13, 248046, 198, 248045, 74455, 198};
 
 void require(bool ok, const std::string& message) {
     if (!ok) throw std::runtime_error(message);
 }
 
 std::string env(const char* name) {
     const char* value = std::getenv(name);
     return value == nullptr ? std::string{} : std::string(value);
 }
 
 void cuda_check(cudaError_t error, const char* operation) {
     if (error != cudaSuccess) {
         throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
     }
 }
 
 class DeviceBuffer {
 public:
     explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
         cuda_check(cudaMalloc(&ptr_, bytes), "cudaMalloc device buffer");
     }
     ~DeviceBuffer() { if (ptr_ != nullptr) cudaFree(ptr_); }
     DeviceBuffer(const DeviceBuffer&) = delete;
     DeviceBuffer& operator=(const DeviceBuffer&) = delete;
     void* get() const noexcept { return ptr_; }
 private:
     void* ptr_ = nullptr;
     std::size_t bytes_ = 0;
 };
 
 int argmax_host(const std::vector<float>& values) {
     require(!values.empty(), "argmax on empty logits");
     return static_cast<int>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
 }
 
 std::string digest_str(const std::array<std::uint64_t, 5>& digest) {
     std::ostringstream out;
     out << std::hex << std::setfill((char)48);
     for (std::size_t i = 0; i < digest.size(); ++i) {
         if (i > 0) out << "-";
         out << std::setw(16) << digest[i];
     }
     return out.str();
 }
 
 // One greedy authoritative target step. Returns the chosen token id.
 std::int64_t greedy_step(Exl3TextContext& ctx) {
     const auto logits = ctx.logits_host();
     const auto token = static_cast<std::int64_t>(argmax_host(logits));
     ctx.decode(token);
     return token;
 }
 
 std::vector<std::int64_t> make_block(std::int64_t anchor, int len) {
     std::vector<std::int64_t> block(static_cast<std::size_t>(len), kMaskToken);
     block[0] = anchor;
     return block;
 }
 
 void require_valid_proposals(const std::vector<std::int64_t>& proposals, int expect_rows,
                              const char* what) {
     require(proposals.size() == static_cast<std::size_t>(expect_rows),
             std::string(what) + ": proposal count mismatch");
     for (const auto token : proposals) {
         require(token >= 0 && token < kVocab, std::string(what) + ": proposal outside vocabulary");
     }
     require(std::any_of(proposals.begin(), proposals.end(),
                         [](std::int64_t token) { return token != 0; }),
             std::string(what) + ": proposals degenerate (all token 0)");
 }
 
 } // namespace
 int main() {
     try {
         const auto target_path = env("NINFER_EXL3_TARGET_PATH");
         const auto draft_path = env("NINFER_EXL3_DFLASH2_PATH");
         if (target_path.empty() || draft_path.empty()) {
             std::cerr << "E5A3 skipped: set NINFER_EXL3_TARGET_PATH and NINFER_EXL3_DFLASH2_PATH\n";
             return 77;
         }
         const std::string results_dir = "results/dflash2";
         std::filesystem::create_directories(results_dir);
         std::ofstream transitions_csv(results_dir + "/E5A3_STATE_TRANSITIONS.csv");
         transitions_csv << "case,accept,committed_rows,ring_base,ring_count,digest,replay_digest,proposal_match,replay_proposal_match\n";
         std::ofstream vram_csv(results_dir + "/E5A3_DRAFT_STATE_VRAM.csv");
         vram_csv << "stage,free_bytes,total_bytes\n";
         std::ofstream latency_csv(results_dir + "/E5A3_PROPOSE_LATENCY_BREAKDOWN.csv");
         latency_csv << "path,wall_median_us,notes\n";
         cuda_check(cudaSetDevice(0), "E5A3 set device");
         {
             void* warm = nullptr;
             cuda_check(cudaMalloc(&warm, 1u << 20u), "E5A3 warmup alloc");
             cuda_check(cudaFree(warm), "E5A3 warmup free");
         }
         cuda_check(cudaDeviceSynchronize(), "E5A3 warmup sync");
         std::size_t free_b = 0, total_b = 0;
         cuda_check(cudaMemGetInfo(&free_b, &total_b), "E5A3 meminfo base");
         vram_csv << "base," << free_b << "," << total_b << "\n";
         auto target = Exl3TextModel::load(target_path, 256);
         auto draft = Exl3Dflash2DraftModel::load(draft_path);
         cuda_check(cudaDeviceSynchronize(), "E5A3 load sync");
         cuda_check(cudaMemGetInfo(&free_b, &total_b), "E5A3 meminfo loaded");
         vram_csv << "target_plus_draft," << free_b << "," << total_b << "\n";
         std::cout << "E5A3 draft weight_bytes=" << draft->weight_bytes()
                   << " scratch_bytes=" << draft->scratch_bytes()
                   << " kv_bytes=" << draft->kv_bytes()
                   << " ring_bytes=" << draft->ring_bytes() << "\n";
         require(draft->ring_bytes() == static_cast<std::size_t>(5) * 2 * 2048 * 1024 * 2 + 2048 * 8,
                 "E5A3 ring byte model mismatch (expect 5*2*2048*1024*2 + 2048*8)");
         std::cout << "E5A3 ring_byte_model=PASS ring_bytes=" << draft->ring_bytes() << "\n";
         // ================= S1: commit + propose_cached == legacy propose =================
         const int P = static_cast<int>(kPrompt.size());  // 14
         // Device staging: prefill bulk (5 x P rows) + single-row slots.
         std::vector<std::unique_ptr<DeviceBuffer>> bulk_storage;
         std::vector<std::uint16_t*> bulk_ptrs;
         for (int t = 0; t < kTapCount; ++t) {
             bulk_storage.push_back(std::make_unique<DeviceBuffer>(
                 static_cast<std::size_t>(P) * kHidden * sizeof(std::uint16_t)));
             bulk_ptrs.push_back(static_cast<std::uint16_t*>(bulk_storage.back()->get()));
         }
         std::vector<std::unique_ptr<DeviceBuffer>> row_storage;
         std::vector<std::uint16_t*> row_ptrs;
         for (int t = 0; t < kTapCount; ++t) {
             row_storage.push_back(std::make_unique<DeviceBuffer>(
                 static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t)));
             row_ptrs.push_back(static_cast<std::uint16_t*>(row_storage.back()->get()));
         }
         auto tap_ctx = target->create_context(true);
         tap_ctx->prefill(kPrompt);
         require(tap_ctx->captured_tap_rows() == P, "E5A3 prefill tap row count mismatch");
         for (int t = 0; t < kTapCount; ++t) {
             tap_ctx->copy_tap_rows_to_device(kTapLayers[static_cast<std::size_t>(t)], 0, bulk_ptrs[t], P);
         }
         cuda_check(cudaDeviceSynchronize(), "E5A3 bulk stage sync");
         draft->reset();
         draft->commit_target_block(bulk_ptrs.data(), P, 0);
         require(draft->ring_count() == P && draft->ring_base_abs() == 0, "E5A3 ring span after prefill commit");
         const auto digest_prefill = draft->ring_digest();
         std::cout << "E5A3 prefill_commit count=" << draft->ring_count()
                   << " base=" << draft->ring_base_abs()
                   << " digest=" << digest_str(digest_prefill) << "\n";
         // Legacy propose over the FULL 14-row window (identical visible K/V).
         Exl3Dflash2TapHistory history(32);
         history.capture_prefill(*tap_ctx, P);
         const auto anchor_win = greedy_step(*tap_ctx);
         history.capture_decode(*tap_ctx);
         // The legacy window is the P prefill rows before the anchor (copy_window
         // excludes the newest row by construction). The ring already holds exactly
         // those rows from the prefill commit above; re-stage through history to
         // prove the history path carries identical values.
         const std::int64_t ctx_pos0 = history.copy_window(*tap_ctx, P, bulk_ptrs.data());
         require(ctx_pos0 == 0, "E5A3 window position mismatch");
         const auto block14 = make_block(anchor_win, kBlockLen);
         const std::vector<std::int64_t> legacy = draft->propose(
             block14, P, bulk_ptrs.data(), P, static_cast<int>(ctx_pos0), P,
             tap_ctx->target_embedding(), tap_ctx->target_lm_head_weights(),
             tap_ctx->target_lm_head_metadata(), kMaskToken);
         require_valid_proposals(legacy, kBlockLen - 1, "E5A3 legacy");
         const std::vector<std::int64_t> cached = draft->propose_cached(
             block14, P, tap_ctx->target_embedding(), tap_ctx->target_lm_head_weights(),
             tap_ctx->target_lm_head_metadata(), kMaskToken);
         require_valid_proposals(cached, kBlockLen - 1, "E5A3 cached");
         require(cached == legacy, "E5A3 ring path diverges from legacy propose on identical context");
         std::array<std::int64_t,kBlockLen> fixed_block{};
         std::copy(block14.begin(),block14.end(),fixed_block.begin());
         const auto fixed_before=fixed_block;
         const auto span_ring_before=draft->ring_digest();
         const auto viewed=draft->propose_cached_view(fixed_block,P,tap_ctx->target_embedding(),
             tap_ctx->target_lm_head_weights(),tap_ctx->target_lm_head_metadata(),kMaskToken);
         require(viewed==cached,"cached span caller changed proposals");
         require(fixed_block==fixed_before,"cached span caller changed borrowed input");
         require(draft->ring_digest()==span_ring_before,"cached span caller mutated committed ring");
         for(std::size_t count:{std::size_t(0),std::size_t(1),static_cast<std::size_t>(draft->block_capacity()+1)}) {
             std::vector<std::int64_t> invalid(count,kMaskToken);
             bool refused=false;
             try{(void)draft->propose_cached_view(invalid,P,tap_ctx->target_embedding(),
                 tap_ctx->target_lm_head_weights(),tap_ctx->target_lm_head_metadata(),kMaskToken);}
             catch(const std::exception& error){refused=std::string(error.what())=="E5A2 block length outside E5A2 capacity";}
             require(refused,"cached span accepted invalid extent");
         }
         for(std::size_t row:{std::size_t(0),std::size_t(7)})for(std::int64_t token:{std::int64_t(-1),std::int64_t(248320)}) {
             auto invalid=fixed_block;invalid[row]=token;
             bool refused=false;
             try{(void)draft->propose_cached_view(invalid,P,tap_ctx->target_embedding(),
                 tap_ctx->target_lm_head_weights(),tap_ctx->target_lm_head_metadata(),kMaskToken);}
             catch(const std::exception& error){refused=std::string(error.what())=="E5A2 draft block token outside vocabulary";}
             require(refused,"cached span accepted invalid token");
         }
         require(draft->ring_digest()==span_ring_before,"invalid cached span changed committed ring");
         for(int position:{-1,2147483647}) {
             bool refused=false;
             try{(void)draft->propose_cached_view(fixed_block,position,tap_ctx->target_embedding(),
                 tap_ctx->target_lm_head_weights(),tap_ctx->target_lm_head_metadata(),kMaskToken);}
             catch(const std::exception& error){refused=std::string(error.what())=="E5A2 draft block position overflow";}
             require(refused,"cached span accepted invalid absolute block position");
         }
         require(draft->ring_digest()==span_ring_before,"invalid cached position changed committed ring");
         for(int context_start:{-1,2147483647,2147483647-P+1}) {
             bool refused=false;
             try{(void)draft->propose(block14,P,bulk_ptrs.data(),P,context_start,P,
                 tap_ctx->target_embedding(),tap_ctx->target_lm_head_weights(),
                 tap_ctx->target_lm_head_metadata(),kMaskToken);}
             catch(const std::exception& error){refused=std::string(error.what())=="E5A2 draft context position overflow";}
             require(refused,"legacy draft accepted invalid context end");
         }
         // End==INT_MAX is representable, but cannot match this fixture's P
         // block position. It must fail lineage, not an overflow check.
         bool boundary_refused=false;
         try{(void)draft->propose(block14,P,bulk_ptrs.data(),P,2147483647-P,P,
             tap_ctx->target_embedding(),tap_ctx->target_lm_head_weights(),
             tap_ctx->target_lm_head_metadata(),kMaskToken);}
         catch(const std::exception& error){boundary_refused=std::string(error.what())==
             "E5A2 draft block must immediately follow the context window";}
         require(boundary_refused,"legacy context end boundary misclassified");
         require(draft->ring_digest()==span_ring_before,"invalid legacy context changed committed ring");
         std::cout << "E5A3 equivalence_ring_vs_legacy=PASS proposals=[";
         for (std::size_t i = 0; i < cached.size(); ++i) {
             if (i > 0) std::cout << ",";
             std::cout << cached[i];
         }
         std::cout << "]\n";
         // Bulk-vs-incremental: reset, commit the same 14 rows one by one.
         draft->reset();
         for (int r = 0; r < P; ++r) {
             // bulk_ptrs holds the contiguous prefill rows [0,14); point at row r.
             for (int t = 0; t < kTapCount; ++t) {
                 row_ptrs[t] = bulk_ptrs[t] + static_cast<std::size_t>(r) * kHidden;
             }
             draft->commit_target_block(row_ptrs.data(), 1, r);
         }
         require(draft->ring_count() == P && draft->ring_base_abs() == 0, "E5A3 ring span after incremental commit");
         const auto digest_incremental = draft->ring_digest();
         require(digest_incremental == digest_prefill, "E5A3 bulk-vs-incremental digest mismatch");
         std::cout << "E5A3 bulk_vs_incremental=PASS\n";
         // Propose-does-not-mutate: digest stable across proposes on both paths.
         const auto before_propose = draft->ring_digest();
         (void)draft->propose_cached(block14, P, tap_ctx->target_embedding(),
                                     tap_ctx->target_lm_head_weights(),
                                     tap_ctx->target_lm_head_metadata(), kMaskToken);
         require(draft->ring_digest() == before_propose, "E5A3 cached propose mutated ring");
         (void)draft->propose(block14, P, bulk_ptrs.data(), P, static_cast<int>(ctx_pos0), P,
                              tap_ctx->target_embedding(), tap_ctx->target_lm_head_weights(),
                              tap_ctx->target_lm_head_metadata(), kMaskToken);
         require(draft->ring_digest() == before_propose, "E5A3 legacy propose mutated ring");
         std::cout << "E5A3 propose_no_mutation=PASS\n";
         // ================= S2: forced accept 0/1/partial/full vs clean replay =================
         const std::array<int, 4> accept_cases = {0, 1, 3, 7};

         for (const int accepted : accept_cases) {
             const int fresh = accepted + 1;  // accepted proposals + 1 target bonus token
             auto ctx_b = target->create_context(true);
             Exl3Dflash2TapHistory history_b(128);
             ctx_b->prefill(kPrompt);
             history_b.capture_prefill(*ctx_b, P);
             std::array<std::vector<float>, kTapCount> prefill_host{};
             for (int t = 0; t < kTapCount; ++t) {
                 prefill_host[t] = ctx_b->hidden_host(kTapLayers[static_cast<std::size_t>(t)]);
                 ctx_b->copy_tap_rows_to_device(kTapLayers[static_cast<std::size_t>(t)], 0, bulk_ptrs[t], P);
             }
             draft->reset();
             draft->commit_target_block(bulk_ptrs.data(), P, 0);
             const auto block_b = make_block(kPrompt.back(), kBlockLen);
             const auto prop_b = draft->propose_cached(
                 block_b, P, ctx_b->target_embedding(), ctx_b->target_lm_head_weights(),
                 ctx_b->target_lm_head_metadata(), kMaskToken);
             require_valid_proposals(prop_b, kBlockLen - 1, "E5A3 transition propose");
             std::vector<std::int64_t> new_tokens;
             new_tokens.reserve(static_cast<std::size_t>(fresh));
             for (int i = 0; i < fresh; ++i) {
                 const auto tok = greedy_step(*ctx_b);
                 new_tokens.push_back(tok);
                 history_b.capture_decode(*ctx_b);
                 for (int t = 0; t < kTapCount; ++t) {
                     ctx_b->copy_tap_row_to_device(kTapLayers[static_cast<std::size_t>(t)], row_ptrs[t]);
                 }
                 draft->commit_target_block(row_ptrs.data(), 1, P + i);
             }
             require(draft->ring_count() == P + fresh && draft->ring_base_abs() == 0,
                     "E5A3 ring span after transition commits");
             const auto anchor2 = new_tokens.back();
             const auto block_b2 = make_block(anchor2, kBlockLen);
             const auto prop_b2 = draft->propose_cached(
                 block_b2, P + fresh, ctx_b->target_embedding(), ctx_b->target_lm_head_weights(),
                 ctx_b->target_lm_head_metadata(), kMaskToken);
             require_valid_proposals(prop_b2, kBlockLen - 1, "E5A3 post-transition propose");
             const auto digest_live = draft->ring_digest();
             // Same-values replay (transition mechanics, exact): restage the identical
             // tap rows from this case history (copy_window excludes the newest row,
             // so capture one extra decode first) and bulk-commit in one call.
             { const auto extra = greedy_step(*ctx_b); (void)extra; history_b.capture_decode(*ctx_b); }
             const int total_c = P + fresh;
             std::vector<std::unique_ptr<DeviceBuffer>> replay_storage;
             std::vector<std::uint16_t*> replay_ptrs;
             for (int t = 0; t < kTapCount; ++t) {
                 replay_storage.push_back(std::make_unique<DeviceBuffer>(
                     static_cast<std::size_t>(total_c) * kHidden * sizeof(std::uint16_t)));
                 replay_ptrs.push_back(static_cast<std::uint16_t*>(replay_storage.back()->get()));
             }
             const std::int64_t replay_pos0 = history_b.copy_window(*ctx_b, total_c, replay_ptrs.data());
             require(replay_pos0 == 0, "E5A3 replay window position mismatch");
             draft->reset();
             draft->commit_target_block(replay_ptrs.data(), total_c, 0);
             const auto digest_replay = draft->ring_digest();
             require(digest_replay == digest_live, "E5A3 transition vs same-values replay digest mismatch");
             const auto prop_c2 = draft->propose_cached(
                 block_b2, total_c, ctx_b->target_embedding(), ctx_b->target_lm_head_weights(),
                 ctx_b->target_lm_head_metadata(), kMaskToken);
             require(prop_c2 == prop_b2, "E5A3 transition vs replay proposal mismatch");
             // Cross-context mirror: fresh context replays the identical token ids
             // with its own commits. Gates token-level stability under the known
             // E5A2 tap-noise regime (rel_l2 ~1e-3) plus greedy target determinism.
             auto ctx_c = target->create_context(true);
             ctx_c->prefill(kPrompt);
             // hidden_host() exposes the last forward only: snapshot the mirror
             // prefill rows here (P rows) for the cross-context noise check below.
             std::array<std::vector<float>, kTapCount> mirror_prefill_host{};
             for (int t = 0; t < kTapCount; ++t) {
                 mirror_prefill_host[t] = ctx_c->hidden_host(kTapLayers[static_cast<std::size_t>(t)]);
             }
             for (int t = 0; t < kTapCount; ++t) {
                 ctx_c->copy_tap_rows_to_device(kTapLayers[static_cast<std::size_t>(t)], 0, replay_ptrs[t], P);
             }
             draft->reset();
             draft->commit_target_block(replay_ptrs.data(), P, 0);
             for (int i = 0; i < fresh; ++i) {
                 const auto tok = greedy_step(*ctx_c);
                 require(tok == new_tokens[static_cast<std::size_t>(i)], "E5A3 cross-context target token mismatch");
                 for (int t = 0; t < kTapCount; ++t) {
                     ctx_c->copy_tap_row_to_device(kTapLayers[static_cast<std::size_t>(t)], row_ptrs[t]);
                 }
                 draft->commit_target_block(row_ptrs.data(), 1, P + i);
             }
             const auto prop_cc = draft->propose_cached(
                 block_b2, total_c, ctx_c->target_embedding(), ctx_c->target_lm_head_weights(),
                 ctx_c->target_lm_head_metadata(), kMaskToken);
            // Cross-context mirror: the E5A3 deterministic split reduction makes
            // identical contexts bit-identical, so the mirror proposal must match
            // exactly (vectors printed for forensics). Same-values replay above is
            // the transition-mechanics gate; this is the end-to-end determinism gate.
            int xctx_agree = 0;
            for (std::size_t pi = 0; pi < prop_cc.size() && pi < prop_b2.size(); ++pi) {
                if (prop_cc[pi] == prop_b2[pi]) ++xctx_agree;
            }
            std::cout << "E5A3 xctx_proposals_live=[";
            for (std::size_t pi = 0; pi < prop_b2.size(); ++pi) {
                if (pi > 0) std::cout << ",";
                std::cout << prop_b2[pi];
            }
            std::cout << "] mirror=[";
            for (std::size_t pi = 0; pi < prop_cc.size(); ++pi) {
                if (pi > 0) std::cout << ",";
                std::cout << prop_cc[pi];
            }
            std::cout << "] agree=" << xctx_agree << "/" << prop_b2.size() << "\n";
            require(draft->ring_count() == total_c && draft->ring_base_abs() == 0,
                    "E5A3 cross-context mirror span mismatch");
            // Hard end-to-end determinism gate (see comment above).
            int first_div = -1;
            for (std::size_t pi = 0; pi < prop_cc.size() && pi < prop_b2.size(); ++pi) {
                if (prop_cc[pi] != prop_b2[pi]) { first_div = static_cast<int>(pi); break; }
            }
            std::cout << "E5A3 xctx_first_div=" << first_div << " agree=" << xctx_agree
                      << "/" << prop_b2.size() << "\n";
            require(prop_cc == prop_b2, "E5A3 cross-context proposal mismatch");
             if (accepted == 0) {
                 for (int t = 0; t < kTapCount; ++t) {
                     const auto& now = mirror_prefill_host[static_cast<std::size_t>(t)];
                     const auto& ref = prefill_host[static_cast<std::size_t>(t)];
                     require(now.size() >= ref.size(), "E5A3 noise check size mismatch");
                     double se = 0.0, sr = 0.0, ma = 0.0;
                     for (std::size_t i = 0; i < ref.size(); ++i) {
                         const double e = static_cast<double>(now[i]) - ref[i];
                         se += e * e; sr += static_cast<double>(ref[i]) * ref[i];
                         ma = std::max(ma, std::fabs(e));
                     }
                     const double rel = sr > 0.0 ? std::sqrt(se / sr) : std::sqrt(se);
                     std::cout << "E5A3 tap_noise_layer_" << kTapLayers[static_cast<std::size_t>(t)]
                               << " max_abs=" << ma << " rel_l2=" << rel << "\n";
                     require(rel <= 0.02, "E5A3 cross-context tap noise above E5A2 regime");
                 }
             }
             std::cout << "E5A3 transition accept=" << accepted << " PASS rows=" << total_c
                       << " digest=" << digest_str(digest_live) << "\n";
             transitions_csv << "transition," << accepted << "," << total_c << ","
                             << draft->ring_base_abs() << "," << draft->ring_count() << ","
                             << digest_str(digest_live) << "," << digest_str(digest_replay)
                             << ",1,1\n";

             // Restore the live span for the next case (reset happens at loop top).
         }

         // Explicit logical rollback: recommit prefill+4, rewind to P, compare with
         // the prefill-only digest, then recommit the same 4 rows and compare with k=3.
         {
             auto ctx_r = target->create_context(true);
             Exl3Dflash2TapHistory history_r(32);
             ctx_r->prefill(kPrompt);
             history_r.capture_prefill(*ctx_r, P);
             for (int t = 0; t < kTapCount; ++t) {
                 ctx_r->copy_tap_rows_to_device(kTapLayers[static_cast<std::size_t>(t)], 0, bulk_ptrs[t], P);
             }
             draft->reset();
             draft->commit_target_block(bulk_ptrs.data(), P, 0);
             const auto digest_r_pre = draft->ring_digest();
             std::vector<std::int64_t> redo;
             for (int i = 0; i < 4; ++i) {
                 const auto tok = greedy_step(*ctx_r);
                 redo.push_back(tok);
                 history_r.capture_decode(*ctx_r);
                 for (int t = 0; t < kTapCount; ++t) {
                     ctx_r->copy_tap_row_to_device(kTapLayers[static_cast<std::size_t>(t)], row_ptrs[t]);
                 }
                 draft->commit_target_block(row_ptrs.data(), 1, P + i);
             }
             { const auto extra = greedy_step(*ctx_r); (void)extra; history_r.capture_decode(*ctx_r); }
             require(draft->ring_count() == P + 4, "E5A3 rollback setup span");
             const auto digest_setup = draft->ring_digest();
             draft->rewind_to(P);
             require(draft->ring_count() == P && draft->ring_base_abs() == 0, "E5A3 rewind span");
             require(draft->ring_digest() == digest_r_pre, "E5A3 rewind digest mismatch");
             std::vector<std::unique_ptr<DeviceBuffer>> redo_storage;
             std::vector<std::uint16_t*> redo_ptrs;
             for (int t = 0; t < kTapCount; ++t) {
                 redo_storage.push_back(std::make_unique<DeviceBuffer>(
                     static_cast<std::size_t>(P + 4) * kHidden * sizeof(std::uint16_t)));
                 redo_ptrs.push_back(static_cast<std::uint16_t*>(redo_storage.back()->get()));
             }
             require(history_r.copy_window(*ctx_r, P + 4, redo_ptrs.data()) == 0,
                     "E5A3 rollback window position mismatch");
             for (int i = 0; i < 4; ++i) {
                 for (int t = 0; t < kTapCount; ++t) {
                     row_ptrs[t] = redo_ptrs[t] + static_cast<std::size_t>(P + i) * kHidden;
                 }
                 draft->commit_target_block(row_ptrs.data(), 1, P + i);
             }
             require(draft->ring_digest() == digest_setup, "E5A3 recommit digest mismatch");
             std::cout << "E5A3 logical_rollback=PASS rewind_to=" << P << " recommit=4\n";
             transitions_csv << "rollback,3," << (P + 4) << ",0," << (P + 4) << ","
                             << digest_str(digest_setup) << "," << digest_str(digest_setup) << ",1,1\n";
         }
         // ================= S3: window rollover + full ring wrap =================
         // Cyclic real-value schedule: pool rows cycle every P positions, so K/V
         // magnitudes are genuine while absolute positions (and RoPE) advance.
         // Commit 1 row per call by pointing into pool rows (no copies).
         const int kRolloverTarget = 2060;
         std::vector<std::uint16_t*> cyc_ptrs(kTapCount);
         draft->reset();
         auto ctx_pool = target->create_context(true);
         ctx_pool->prefill(kPrompt);
         for (int t = 0; t < kTapCount; ++t) {
             ctx_pool->copy_tap_rows_to_device(kTapLayers[static_cast<std::size_t>(t)], 0, bulk_ptrs[t], P);
         }
         cuda_check(cudaDeviceSynchronize(), "E5A3 pool stage sync");
         for (int p = 0; p < kRolloverTarget; ++p) {
             for (int t = 0; t < kTapCount; ++t) {
                 cyc_ptrs[t] = bulk_ptrs[t] +
                     static_cast<std::size_t>(p % P) * kHidden;
             }
             draft->commit_target_block(cyc_ptrs.data(), 1, p);
             if (p + 1 == 2047) {
                 require(draft->ring_count() == 2047 && draft->ring_base_abs() == 0,
                         "E5A3 window-1 span");
                 std::cout << "E5A3 rollover window_minus_1 PASS count=2047 base=0\n";
             }
             if (p + 1 == 2048) {
                 require(draft->ring_count() == 2047 && draft->ring_base_abs() == 1,
                         "E5A3 window span (first eviction)");
                 std::cout << "E5A3 rollover window PASS count=2047 base=1\n";
             }
             if (p + 1 == 2049) {
                 require(draft->ring_count() == 2047 && draft->ring_base_abs() == 2,
                         "E5A3 window+1 span");
                 std::cout << "E5A3 rollover window_plus_1 PASS count=2047 base=2\n";
             }
         }
         require(draft->ring_count() == 2047 && draft->ring_base_abs() == 13,
                 "E5A3 rollover 2060 span");
         const auto digest_2060 = draft->ring_digest();
         std::cout << "E5A3 rollover rows=2060 PASS base=13 digest=" << digest_str(digest_2060) << "\n";
         const auto block_wrap = make_block(42, kBlockLen);
         const auto prop_wrap = draft->propose_cached(
             block_wrap, kRolloverTarget, tap_ctx->target_embedding(),
             tap_ctx->target_lm_head_weights(), tap_ctx->target_lm_head_metadata(), kMaskToken);
         require_valid_proposals(prop_wrap, kBlockLen - 1, "E5A3 wrap propose");
         const auto prop_wrap2 = draft->propose_cached(
             block_wrap, kRolloverTarget, tap_ctx->target_embedding(),
             tap_ctx->target_lm_head_weights(), tap_ctx->target_lm_head_metadata(), kMaskToken);
         require(prop_wrap2 == prop_wrap, "E5A3 wrap propose not deterministic");
         require(draft->ring_digest() == digest_2060, "E5A3 propose mutated full ring");
         std::cout << "E5A3 wrap_propose=PASS deterministic finite\n";
         transitions_csv << "wrap_propose,7," << kRolloverTarget << ",13,2047,"
                         << digest_str(digest_2060) << "," << digest_str(digest_2060) << ",1,1\n";
         // Bulk prefill-skip (>2047 rows on an empty ring) must equal incremental.
         const int kBulkRows = 2060;
         std::vector<std::unique_ptr<DeviceBuffer>> bulk2_storage;
         std::vector<std::uint16_t*> bulk2_ptrs;
         for (int t = 0; t < kTapCount; ++t) {
             bulk2_storage.push_back(std::make_unique<DeviceBuffer>(
                 static_cast<std::size_t>(kBulkRows) * kHidden * sizeof(std::uint16_t)));
             bulk2_ptrs.push_back(static_cast<std::uint16_t*>(bulk2_storage.back()->get()));
         }
         for (int r = 0; r < kBulkRows; ++r) {
             for (int t = 0; t < kTapCount; ++t) {
                 cuda_check(cudaMemcpy(bulk2_ptrs[t] + static_cast<std::size_t>(r) * kHidden,
                                       bulk_ptrs[t] + static_cast<std::size_t>(r % P) * kHidden,
                                       static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToDevice),
                            "E5A3 cyclic bulk fill");
             }
         }
         draft->reset();
         draft->commit_target_block(bulk2_ptrs.data(), kBulkRows, 0);
         require(draft->ring_count() == 2047 && draft->ring_base_abs() == 13,
                 "E5A3 bulk-skip span (expect base=13 count=2047)");
         require(draft->ring_digest() == digest_2060, "E5A3 bulk-skip vs incremental digest mismatch");
         std::cout << "E5A3 bulk_skip=PASS base=13 count=2047\n";
         // Full wrap: continue incremental commits to 4150 (past one complete reuse).
         for (int p = kBulkRows; p < 4150; ++p) {
             for (int t = 0; t < kTapCount; ++t) {
                 cyc_ptrs[t] = bulk_ptrs[t] +
                     static_cast<std::size_t>(p % P) * kHidden;
             }
             draft->commit_target_block(cyc_ptrs.data(), 1, p);
         }
         require(draft->ring_count() == 2047 && draft->ring_base_abs() == 2103,
                 "E5A3 full-wrap span (expect base=2103 count=2047)");
         cuda_check(cudaMemGetInfo(&free_b, &total_b), "E5A3 meminfo after wrap");
         vram_csv << "after_wrap," << free_b << "," << total_b << "\n";
         const auto digest_wrap = draft->ring_digest();
         const auto prop_full = draft->propose_cached(
             block_wrap, 4150, tap_ctx->target_embedding(),
             tap_ctx->target_lm_head_weights(), tap_ctx->target_lm_head_metadata(), kMaskToken);
         require_valid_proposals(prop_full, kBlockLen - 1, "E5A3 full-wrap propose");
         const auto prop_full2 = draft->propose_cached(
             block_wrap, 4150, tap_ctx->target_embedding(),
             tap_ctx->target_lm_head_weights(), tap_ctx->target_lm_head_metadata(), kMaskToken);
         require(prop_full2 == prop_full, "E5A3 wrapped-ring propose not deterministic");
         require(draft->ring_digest() == digest_wrap, "E5A3 propose mutated wrapped ring");
         std::cout << "E5A3 full_wrap=PASS rows=4150 base=2103 digest=" << digest_str(digest_wrap) << "\n";
         transitions_csv << "full_wrap,7,4150,2103,2047,"
                         << digest_str(digest_wrap) << "," << digest_str(digest_wrap) << ",1,1\n";
         // ================= S4: reset + destroy/recreate =================
         draft->reset();
         require(draft->ring_count() == 0 && draft->ring_base_abs() == 0, "E5A3 reset after wrap");
         auto ctx_s = target->create_context(true);
         ctx_s->prefill(kPrompt);
         for (int t = 0; t < kTapCount; ++t) {
             ctx_s->copy_tap_rows_to_device(kTapLayers[static_cast<std::size_t>(t)], 0, bulk_ptrs[t], P);
         }
         draft->commit_target_block(bulk_ptrs.data(), P, 0);
         const auto digest_s4 = draft->ring_digest();
         std::cout << "E5A3 reset_after_wrap=PASS\n";
         {
             std::size_t free_before_destroy = 0, total_tmp = 0;
             cuda_check(cudaMemGetInfo(&free_before_destroy, &total_tmp), "E5A3 meminfo before destroy");
             draft.reset();
             cuda_check(cudaDeviceSynchronize(), "E5A3 destroy sync");
             std::size_t free_after_destroy = 0;
             cuda_check(cudaMemGetInfo(&free_after_destroy, &total_tmp), "E5A3 meminfo after destroy");
             vram_csv << "draft_destroyed," << free_after_destroy << "," << total_tmp << "\n";
             std::cout << "E5A3 destroy freed_bytes=" << (free_after_destroy - free_before_destroy) << "\n";
             draft = Exl3Dflash2DraftModel::load(draft_path);
             cuda_check(cudaDeviceSynchronize(), "E5A3 recreate sync");
             std::size_t free_after_recreate = 0;
             cuda_check(cudaMemGetInfo(&free_after_recreate, &total_tmp), "E5A3 meminfo after recreate");
             vram_csv << "draft_recreated," << free_after_recreate << "," << total_tmp << "\n";
             for (int t = 0; t < kTapCount; ++t) {
                 ctx_s->copy_tap_rows_to_device(kTapLayers[static_cast<std::size_t>(t)], 0, bulk_ptrs[t], P);
             }
             draft->commit_target_block(bulk_ptrs.data(), P, 0);
             require(draft->ring_digest() == digest_s4, "E5A3 recreated-model digest mismatch");
             std::cout << "E5A3 destroy_recreate=PASS\n";
         }
         // ================= S5/S6: latency (LAST: enables env-gated phase timers) =================
         history.copy_window(*tap_ctx, P, bulk_ptrs.data());  // restore S1 prefill window [0,14)
         draft->reset();
         draft->commit_target_block(bulk_ptrs.data(), P, 0);
         require(draft->ring_digest() == digest_prefill, "E5A3 S6 window restore mismatch");
         auto event_timed = [&](const std::string& what, bool use_ring) {
             std::vector<double> samples;
             for (int i = 0; i < 6; ++i) {
                 cudaEvent_t start = nullptr, stop = nullptr;
                 cuda_check(cudaEventCreate(&start), "E5A3 latency create start");
                 cuda_check(cudaEventCreate(&stop), "E5A3 latency create stop");
                 cuda_check(cudaEventRecord(start), "E5A3 latency record start");
                 if (use_ring) {
                     (void)draft->propose_cached(block14, P, tap_ctx->target_embedding(),
                                                 tap_ctx->target_lm_head_weights(),
                                                 tap_ctx->target_lm_head_metadata(), kMaskToken);
                 } else {
                     (void)draft->propose(block14, P, bulk_ptrs.data(), P, static_cast<int>(ctx_pos0), P,
                                          tap_ctx->target_embedding(), tap_ctx->target_lm_head_weights(),
                                          tap_ctx->target_lm_head_metadata(), kMaskToken);
                 }
                 cuda_check(cudaEventRecord(stop), "E5A3 latency record stop");
                 cuda_check(cudaEventSynchronize(stop), "E5A3 latency sync");
                 float ms = 0.0f;
                 cuda_check(cudaEventElapsedTime(&ms, start, stop), "E5A3 latency elapsed");
                 cuda_check(cudaEventDestroy(start), "E5A3 latency destroy start");
                 cuda_check(cudaEventDestroy(stop), "E5A3 latency destroy stop");
                 if (i > 0) samples.push_back(static_cast<double>(ms) * 1000.0);
             }
             std::sort(samples.begin(), samples.end());
             const double median = samples[samples.size() / 2];
             std::cout << "E5A3 latency " << what << "_median_us=" << std::fixed << std::setprecision(1)
                       << median << std::defaultfloat << "\n";
             latency_csv << what << "," << median << ",cuda-event device time, block8 window14" << "\n";
             return median;
         };
         const auto wall_start = std::chrono::steady_clock::now();
         const double legacy_us = event_timed("legacy_propose", false);
         const double ring_us = event_timed("ring_propose", true);
         const double batch_wall_us = std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - wall_start).count();
         std::cout << "E5A3 latency_batch_wall_us=" << std::fixed << std::setprecision(0)
                   << batch_wall_us << std::defaultfloat << "\n";
         latency_csv << "batch_wall_12_calls," << batch_wall_us << ",chrono wall for 12 proposes" << "\n";
         (void)legacy_us; (void)ring_us;
         _putenv_s("NINFER_DFLASH2_TIMING", "1");
         (void)draft->propose(block14, P, bulk_ptrs.data(), P, static_cast<int>(ctx_pos0), P,
                              tap_ctx->target_embedding(), tap_ctx->target_lm_head_weights(),
                              tap_ctx->target_lm_head_metadata(), kMaskToken);
         (void)draft->propose_cached(block14, P, tap_ctx->target_embedding(),
                                     tap_ctx->target_lm_head_weights(),
                                     tap_ctx->target_lm_head_metadata(), kMaskToken);
         std::cout << "E5A3 phase_breakdown_printed=PASS (see E5A3 propose_timing lines above)\n";
         std::cout << "E5A3_DRAFT_STATE_QUALIFIED\n";
         return 0;
     } catch (const std::exception& ex) {
         std::cerr << "E5A3 FAILED: " << ex.what() << "\n";
         return 1;
     }
 }
