 // E5A3 diagnostic W: wrap-regime propose determinism.
 // Cyclic real-value commits to 2060 (as S3), then 3 identical proposes with
 // digests; plus a small-ring (100 rows) cyclic control.
 #include "exl3/text_model.h"
 #include "exl3/dflash2_draft.h"

 #include <cuda_runtime.h>

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

 } // namespace

 int main() {
     try {
         const auto target_path = env("NINFER_EXL3_TARGET_PATH");
         const auto draft_path = env("NINFER_EXL3_DFLASH2_PATH");
         if (target_path.empty() || draft_path.empty()) {
             std::cerr << "WRAP skipped: set env paths\n";
             return 77;
         }
         cuda_check(cudaSetDevice(0), "WRAP set device");
         auto target = Exl3TextModel::load(target_path, 256);
         auto draft = Exl3Dflash2DraftModel::load(draft_path);
         const int P = static_cast<int>(kPrompt.size());
         std::vector<std::unique_ptr<DeviceBuffer>> bulk_store;
         std::vector<std::uint16_t*> bulk;
         for (int t = 0; t < kTapCount; ++t) {
             bulk_store.push_back(std::make_unique<DeviceBuffer>(
                 static_cast<std::size_t>(P) * 5120 * sizeof(std::uint16_t)));
             bulk.push_back(static_cast<std::uint16_t*>(bulk_store.back()->get()));
         }
         auto ctx = target->create_context(true);
         ctx->prefill(kPrompt);
         for (int t = 0; t < kTapCount; ++t) {
             ctx->copy_tap_rows_to_device(kTapLayers[static_cast<std::size_t>(t)], 0, bulk[t], P);
         }
         cuda_check(cudaDeviceSynchronize(), "WRAP stage sync");
         std::vector<std::uint16_t*> cyc(kTapCount);
         // Control: small ring, cyclic values, 100 rows.
         draft->reset();
         for (int p = 0; p < 100; ++p) {
             for (int t = 0; t < kTapCount; ++t) {
                 cyc[t] = bulk[t] + static_cast<std::size_t>(p % P) * 5120;
             }
             draft->commit_target_block(cyc.data(), 1, p);
         }
         std::vector<std::int64_t> block(8, kMaskToken);
         block[0] = 42;
         const auto s1 = draft->propose_cached(block, 100, ctx->target_embedding(),
             ctx->target_lm_head_weights(), ctx->target_lm_head_metadata(), kMaskToken);
         const auto s2 = draft->propose_cached(block, 100, ctx->target_embedding(),
             ctx->target_lm_head_weights(), ctx->target_lm_head_metadata(), kMaskToken);
         std::cout << "WRAP small_cyclic s1=" << vec_str(s1) << "\n";
         std::cout << "WRAP small_cyclic s2=" << vec_str(s2) << "\n";
         std::cout << "WRAP small_cyclic_determinism=" << (s1 == s2 ? "PASS" : "FAIL") << "\n";
         // Full regime: continue cyclic commits to 2060.
         for (int p = 100; p < 2060; ++p) {
             for (int t = 0; t < kTapCount; ++t) {
                 cyc[t] = bulk[t] + static_cast<std::size_t>(p % P) * 5120;
             }
             draft->commit_target_block(cyc.data(), 1, p);
         }
         std::cout << "WRAP span count=" << draft->ring_count()
                   << " base=" << draft->ring_base_abs() << "\n";
         const auto d0 = draft->ring_digest();
         const auto w1 = draft->propose_cached(block, 2060, ctx->target_embedding(),
             ctx->target_lm_head_weights(), ctx->target_lm_head_metadata(), kMaskToken);
         const auto d1 = draft->ring_digest();
         const auto w2 = draft->propose_cached_view(block, 2060, ctx->target_embedding(),
             ctx->target_lm_head_weights(), ctx->target_lm_head_metadata(), kMaskToken);
         const auto d2 = draft->ring_digest();
         const auto w3 = draft->propose_cached(block, 2060, ctx->target_embedding(),
             ctx->target_lm_head_weights(), ctx->target_lm_head_metadata(), kMaskToken);
         const auto d3 = draft->ring_digest();
         if(w1!=w2 || w2!=w3 || d0!=d1 || d1!=d2 || d2!=d3)
             throw std::runtime_error("wrapped ring span/vector proposal or state mismatch");
         std::cout << "WRAP w1=" << vec_str(w1) << " digest_same=" << (d1 == d0 ? 1 : 0) << "\n";
         std::cout << "WRAP w2=" << vec_str(w2) << " digest_same=" << (d2 == d0 ? 1 : 0) << "\n";
         std::cout << "WRAP w3=" << vec_str(w3) << " digest_same=" << (d3 == d0 ? 1 : 0) << "\n";
         std::cout << "WRAP full_w1_eq_w2=" << (w1 == w2 ? "PASS" : "FAIL") << "\n";
         std::cout << "WRAP full_w2_eq_w3=" << (w2 == w3 ? "PASS" : "FAIL") << "\n";
         // E2: bulk commit of the same 2060 cyclic rows, then propose x2.
         std::vector<std::unique_ptr<DeviceBuffer>> bulk2_store;
         std::vector<std::uint16_t*> bulk2;
         for (int tt = 0; tt < kTapCount; ++tt) {
             bulk2_store.push_back(std::make_unique<DeviceBuffer>(
                 static_cast<std::size_t>(2060) * 5120 * sizeof(std::uint16_t)));
             bulk2.push_back(static_cast<std::uint16_t*>(bulk2_store.back()->get()));
         }
         for (int r = 0; r < 2060; ++r) {
             for (int tt = 0; tt < kTapCount; ++tt) {
                 cuda_check(cudaMemcpy(bulk2[tt] + static_cast<std::size_t>(r) * 5120,
                                       bulk[tt] + static_cast<std::size_t>(r % P) * 5120,
                                       static_cast<std::size_t>(5120) * sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToDevice),
                            "WRAP cyclic bulk fill");
             }
         }
         draft->reset();
         draft->commit_target_block(bulk2.data(), 2060, 0);
         const auto db = draft->ring_digest();
         std::cout << "WRAP bulk_span count=" << draft->ring_count()
                   << " base=" << draft->ring_base_abs()
                   << " digest_eq_incr=" << (db == d0 ? 1 : 0) << "\n";
         const auto b1 = draft->propose_cached(block, 2060, ctx->target_embedding(),
             ctx->target_lm_head_weights(), ctx->target_lm_head_metadata(), kMaskToken);
         const auto b2 = draft->propose_cached(block, 2060, ctx->target_embedding(),
             ctx->target_lm_head_weights(), ctx->target_lm_head_metadata(), kMaskToken);
         std::cout << "WRAP b1=" << vec_str(b1) << "\n";
         std::cout << "WRAP b2=" << vec_str(b2) << "\n";
         std::cout << "WRAP bulk_b1_eq_b2=" << (b1 == b2 ? "PASS" : "FAIL") << "\n";
         std::cout << "WRAP bulk_b1_eq_w2=" << (b1 == w2 ? "PASS" : "FAIL") << "\n";
         // E3: 5 more incremental commits, then propose x2.
         for (int p = 2060; p < 2065; ++p) {
             for (int tt = 0; tt < kTapCount; ++tt) {
                 cyc[tt] = bulk[tt] + static_cast<std::size_t>(p % P) * 5120;
             }
             draft->commit_target_block(cyc.data(), 1, p);
         }
         const auto c1 = draft->propose_cached(block, 2065, ctx->target_embedding(),
             ctx->target_lm_head_weights(), ctx->target_lm_head_metadata(), kMaskToken);
         const auto c2 = draft->propose_cached(block, 2065, ctx->target_embedding(),
             ctx->target_lm_head_weights(), ctx->target_lm_head_metadata(), kMaskToken);
         std::cout << "WRAP c1=" << vec_str(c1) << "\n";
         std::cout << "WRAP c2=" << vec_str(c2) << "\n";
         std::cout << "WRAP incr5_c1_eq_c2=" << (c1 == c2 ? "PASS" : "FAIL") << "\n";
         std::cout << "WRAP_DONE\n";
         return 0;
     } catch (const std::exception& ex) {
         std::cerr << "WRAP FAILED: " << ex.what() << "\n";
         return 1;
     }
 }
