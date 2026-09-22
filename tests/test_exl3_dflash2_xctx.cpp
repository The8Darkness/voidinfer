 // E5A3 diagnostic: cross-context determinism probe (exit 0 unless exception).
 // Isolates the S2 "cross-context proposal mismatch": compares ring digests and
 // proposals across (a) repeated same-context proposes, (b) fresh-context
 // prefill commit, (c) fresh-context decode commit.
 #include "exl3/text_model.h"
 #include "exl3/dflash2_draft.h"

 #include <cuda_runtime.h>

 #include <algorithm>
 #include <array>
 #include <cstdint>
 #include <cstdlib>
 #include <iomanip>
 #include <iostream>
 #include <sstream>
 #include <stdexcept>
 #include <string>
 #include <vector>

 namespace {

 using ninfer::exl3::Exl3TextModel;
 using ninfer::exl3::Exl3TextContext;
 using ninfer::exl3::Exl3Dflash2DraftModel;

 constexpr int kHidden = 5120;
 constexpr int kTapCount = 5;
 constexpr int64_t kMaskToken = 248070;
 constexpr int kBlockLen = 8;
 constexpr std::array<int, kTapCount> kTapLayers = {5, 19, 33, 47, 61};

 const std::vector<std::int64_t> kPrompt = {248045, 846, 198, 7734, 799, 11316, 883,
                                            12050, 13, 248046, 198, 248045, 74455, 198};

 std::string env(const char* name) {
     const char* value = std::getenv(name);
     return value == nullptr ? std::string{} : std::string(value);
 }

 void cuda_check(cudaError_t error, const char* operation) {
     if (error != cudaSuccess) {
         throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
     }
 }

 struct DeviceBuffer {
     explicit DeviceBuffer(std::size_t bytes) {
         cuda_check(cudaMalloc(&ptr_, bytes), "cudaMalloc device buffer");
     }
     ~DeviceBuffer() { if (ptr_ != nullptr) cudaFree(ptr_); }
     void* get() const noexcept { return ptr_; }
     void* ptr_ = nullptr;
 };

 int argmax_host(const std::vector<float>& values) {
     return static_cast<int>(std::distance(values.begin(),
         std::max_element(values.begin(), values.end())));
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

 std::string vec_str(const std::vector<std::int64_t>& v) {
     std::ostringstream out;
     out << "[";
     for (std::size_t i = 0; i < v.size(); ++i) {
         if (i > 0) out << ",";
         out << v[i];
     }
     out << "]";
     return out.str();
 }

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

 } // namespace

 int main() {
     try {
         const auto target_path = env("NINFER_EXL3_TARGET_PATH");
         const auto draft_path = env("NINFER_EXL3_DFLASH2_PATH");
         if (target_path.empty() || draft_path.empty()) {
             std::cerr << "XCTX skipped: set env paths\n";
             return 77;
         }
         cuda_check(cudaSetDevice(0), "XCTX set device");
         auto target = Exl3TextModel::load(target_path, 256);
         auto draft = Exl3Dflash2DraftModel::load(draft_path);
         const int P = static_cast<int>(kPrompt.size());
         std::vector<std::unique_ptr<DeviceBuffer>> bulk_a_store, bulk_b_store, row_store;
         std::vector<std::uint16_t*> bulk_a, bulk_b, row_ptrs;
         for (int t = 0; t < kTapCount; ++t) {
             bulk_a_store.push_back(std::make_unique<DeviceBuffer>(
                 static_cast<std::size_t>(P) * 5120 * sizeof(std::uint16_t)));
             bulk_a.push_back(static_cast<std::uint16_t*>(bulk_a_store.back()->get()));
             bulk_b_store.push_back(std::make_unique<DeviceBuffer>(
                 static_cast<std::size_t>(P) * 5120 * sizeof(std::uint16_t)));
             bulk_b.push_back(static_cast<std::uint16_t*>(bulk_b_store.back()->get()));
             row_store.push_back(std::make_unique<DeviceBuffer>(
                 static_cast<std::size_t>(5120) * sizeof(std::uint16_t)));
             row_ptrs.push_back(static_cast<std::uint16_t*>(row_store.back()->get()));
         }
         // ---- context A: prefill ----
         auto ctx_a = target->create_context(true);
         ctx_a->prefill(kPrompt);
         for (int t = 0; t < kTapCount; ++t) {
             ctx_a->copy_tap_rows_to_device(kTapLayers[static_cast<std::size_t>(t)], 0, bulk_a[t], P);
         }
         cuda_check(cudaDeviceSynchronize(), "XCTX stage A sync");
         draft->reset();
         draft->commit_target_block(bulk_a.data(), P, 0);
         const auto digest_A_pre = draft->ring_digest();
         std::cout << "XCTX digest_A_pre=" << digest_str(digest_A_pre) << "\n";
         const auto block_a = make_block(kPrompt.back(), kBlockLen);
         const auto pa1 = draft->propose_cached(block_a, P, ctx_a->target_embedding(),
             ctx_a->target_lm_head_weights(), ctx_a->target_lm_head_metadata(), kMaskToken);
         const auto pa2 = draft->propose_cached(block_a, P, ctx_a->target_embedding(),
             ctx_a->target_lm_head_weights(), ctx_a->target_lm_head_metadata(), kMaskToken);
         std::cout << "XCTX pa1=" << vec_str(pa1) << "\n";
         std::cout << "XCTX pa2=" << vec_str(pa2) << "\n";
         std::cout << "XCTX same_context_determinism=" << (pa1 == pa2 ? "PASS" : "FAIL") << "\n";
         // ---- context A: one decode ----
         const auto ta = greedy_step(*ctx_a);
         for (int t = 0; t < kTapCount; ++t) {
             ctx_a->copy_tap_row_to_device(kTapLayers[static_cast<std::size_t>(t)], row_ptrs[t]);
         }
         cuda_check(cudaDeviceSynchronize(), "XCTX stage Adec sync");
         draft->commit_target_block(row_ptrs.data(), 1, P);
         const auto digest_A_dec = draft->ring_digest();
         std::cout << "XCTX token_a=" << ta << " digest_A_dec=" << digest_str(digest_A_dec) << "\n";
         const auto block_a2 = make_block(ta, kBlockLen);
         const auto pa3 = draft->propose_cached(block_a2, P + 1, ctx_a->target_embedding(),
             ctx_a->target_lm_head_weights(), ctx_a->target_lm_head_metadata(), kMaskToken);
         const auto pa4 = draft->propose_cached(block_a2, P + 1, ctx_a->target_embedding(),
             ctx_a->target_lm_head_weights(), ctx_a->target_lm_head_metadata(), kMaskToken);
         std::cout << "XCTX pa3=" << vec_str(pa3) << "\n";
         std::cout << "XCTX same_context_determinism_postdec=" << (pa3 == pa4 ? "PASS" : "FAIL") << "\n";
         // ---- context B: prefill ----
         auto ctx_b = target->create_context(true);
         ctx_b->prefill(kPrompt);
         for (int t = 0; t < kTapCount; ++t) {
             ctx_b->copy_tap_rows_to_device(kTapLayers[static_cast<std::size_t>(t)], 0, bulk_b[t], P);
         }
         cuda_check(cudaDeviceSynchronize(), "XCTX stage B sync");
         draft->reset();
         draft->commit_target_block(bulk_b.data(), P, 0);
         const auto digest_B_pre = draft->ring_digest();
         std::cout << "XCTX digest_B_pre=" << digest_str(digest_B_pre) << "\n";
         std::cout << "XCTX prefill_xctx_digest=" << (digest_A_pre == digest_B_pre ? "PASS" : "FAIL") << "\n";
         const auto pb_pre = draft->propose_cached(block_a, P, ctx_b->target_embedding(),
             ctx_b->target_lm_head_weights(), ctx_b->target_lm_head_metadata(), kMaskToken);
         std::cout << "XCTX pb_pre=" << vec_str(pb_pre) << "\n";
         std::cout << "XCTX prefill_xctx_proposal=" << (pb_pre == pa1 ? "PASS" : "FAIL") << "\n";
         // ---- context B: one decode ----
         const auto tb = greedy_step(*ctx_b);
         std::cout << "XCTX token_b=" << tb << " target_xctx_token=" << (tb == ta ? "PASS" : "FAIL") << "\n";
         for (int t = 0; t < kTapCount; ++t) {
             ctx_b->copy_tap_row_to_device(kTapLayers[static_cast<std::size_t>(t)], row_ptrs[t]);
         }
         cuda_check(cudaDeviceSynchronize(), "XCTX stage Bdec sync");
         draft->commit_target_block(row_ptrs.data(), 1, P);
         const auto digest_B_dec = draft->ring_digest();
         std::cout << "XCTX digest_B_dec=" << digest_str(digest_B_dec) << "\n";
         std::cout << "XCTX decode_xctx_digest=" << (digest_A_dec == digest_B_dec ? "PASS" : "FAIL") << "\n";
         const auto block_b2 = make_block(tb, kBlockLen);
         const auto pb = draft->propose_cached(block_b2, P + 1, ctx_b->target_embedding(),
             ctx_b->target_lm_head_weights(), ctx_b->target_lm_head_metadata(), kMaskToken);
         std::cout << "XCTX pb=" << vec_str(pb) << "\n";
         std::cout << "XCTX decode_xctx_proposal=" << (pb == pa3 ? "PASS" : "FAIL") << "\n";
         std::cout << "XCTX_DONE\n";
         return 0;
     } catch (const std::exception& ex) {
         std::cerr << "XCTX FAILED: " << ex.what() << "\n";
         return 1;
     }
 }
