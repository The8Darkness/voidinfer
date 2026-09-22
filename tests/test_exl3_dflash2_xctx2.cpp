 // E5A3 diagnostic v2: attribute cross-context tap divergence.
 // X1: commit determinism (same staged taps committed twice -> digest equal?).
 // X2: prefill tap byte equality across contexts (host FNV of staged rows).
 // X3: second-decode token + tap-row byte equality across contexts.
 #include "exl3/text_model.h"
 #include "exl3/dflash2_draft.h"

 #include <cuda_runtime.h>

 #include <algorithm>
 #include <array>
 #include <cmath>
 #include <cstdint>
 #include <cstring>
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

 std::uint64_t fnv_bytes(const void* data, std::size_t n) {
     const auto* p = static_cast<const unsigned char*>(data);
     std::uint64_t h = 1469598103934665603ULL;
     for (std::size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
     return h;
 }

 std::int64_t greedy_step(Exl3TextContext& ctx) {
     const auto logits = ctx.logits_host();
     const auto token = static_cast<std::int64_t>(argmax_host(logits));
     ctx.decode(token);
     return token;
 }

 } // namespace

 int main() {
     try {
         const auto target_path = env("NINFER_EXL3_TARGET_PATH");
         const auto draft_path = env("NINFER_EXL3_DFLASH2_PATH");
         if (target_path.empty() || draft_path.empty()) {
             std::cerr << "XCTX2 skipped: set env paths\n";
             return 77;
         }
         cuda_check(cudaSetDevice(0), "XCTX2 set device");
         auto target = Exl3TextModel::load(target_path, 256);
         auto draft = Exl3Dflash2DraftModel::load(draft_path);
         const int P = static_cast<int>(kPrompt.size());
         const std::size_t bulk_bytes = static_cast<std::size_t>(P) * 5120 * sizeof(std::uint16_t);
         std::vector<std::unique_ptr<DeviceBuffer>> bulk_a_store, bulk_b_store, row_store;
         std::vector<std::uint16_t*> bulk_a, bulk_b, row_ptrs;
         for (int t = 0; t < kTapCount; ++t) {
             bulk_a_store.push_back(std::make_unique<DeviceBuffer>(bulk_bytes));
             bulk_a.push_back(static_cast<std::uint16_t*>(bulk_a_store.back()->get()));
             bulk_b_store.push_back(std::make_unique<DeviceBuffer>(bulk_bytes));
             bulk_b.push_back(static_cast<std::uint16_t*>(bulk_b_store.back()->get()));
             row_store.push_back(std::make_unique<DeviceBuffer>(
                 static_cast<std::size_t>(5120) * sizeof(std::uint16_t)));
             row_ptrs.push_back(static_cast<std::uint16_t*>(row_store.back()->get()));
         }
         auto ctx_a = target->create_context(true);
         ctx_a->prefill(kPrompt);
         for (int t = 0; t < kTapCount; ++t) {
             ctx_a->copy_tap_rows_to_device(kTapLayers[static_cast<std::size_t>(t)], 0, bulk_a[t], P);
         }
         auto ctx_b = target->create_context(true);
         ctx_b->prefill(kPrompt);
         for (int t = 0; t < kTapCount; ++t) {
             ctx_b->copy_tap_rows_to_device(kTapLayers[static_cast<std::size_t>(t)], 0, bulk_b[t], P);
         }
         cuda_check(cudaDeviceSynchronize(), "XCTX2 stage sync");
         // X1: commit determinism on identical staged taps.
         draft->reset();
         draft->commit_target_block(bulk_a.data(), P, 0);
         const auto d1 = draft->ring_digest();
         draft->reset();
         draft->commit_target_block(bulk_a.data(), P, 0);
         const auto d2 = draft->ring_digest();
         std::cout << "XCTX2 commit_determinism=" << (d1 == d2 ? "PASS" : "FAIL") << "\n";
         std::cout << "XCTX2 d1=" << digest_str(d1) << "\n";
         std::cout << "XCTX2 d2=" << digest_str(d2) << "\n";
         // X2: staged prefill tap bytes across contexts.
         std::vector<std::uint16_t> host(static_cast<std::size_t>(P) * 5120);
         bool taps_equal = true;
         for (int t = 0; t < kTapCount; ++t) {
             cuda_check(cudaMemcpy(host.data(), bulk_a[t], bulk_bytes, cudaMemcpyDeviceToHost),
                        "XCTX2 download A");
             const std::uint64_t ha = fnv_bytes(host.data(), bulk_bytes);
             cuda_check(cudaMemcpy(host.data(), bulk_b[t], bulk_bytes, cudaMemcpyDeviceToHost),
                        "XCTX2 download B");
             const std::uint64_t hb = fnv_bytes(host.data(), bulk_bytes);
             // max abs diff + rel l2 for the record
             cuda_check(cudaMemcpy(host.data(), bulk_a[t], bulk_bytes, cudaMemcpyDeviceToHost),
                        "XCTX2 redownload A");
             std::vector<std::uint16_t> hostb(host.size());
             cuda_check(cudaMemcpy(hostb.data(), bulk_b[t], bulk_bytes, cudaMemcpyDeviceToHost),
                        "XCTX2 redownload B");
             double se = 0.0, sr = 0.0;
             int diff_words = 0;
             for (std::size_t i = 0; i < host.size(); ++i) {
                 if (host[i] != hostb[i]) ++diff_words;
                 float fa, fb;
                 std::uint16_t xa = host[i], xb = hostb[i];
                 std::memcpy(&fa, &xa, 2);
                 // F16 bits -> float via host conversion is overkill; compare u16-mapped magnitudes
                 // using __half? Keep it simple: report word diffs + FNV only, plus F16-decoded stats
                 // through a portable half->float.
                 auto half_to_float = [](std::uint16_t h) -> double {
                     int s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
                     double v;
                     if (e == 0) v = (m / 1024.0) * std::pow(2.0, -14);
                     else if (e == 31) v = (m == 0) ? 1e30 : 1e30;
                     else v = (1.0 + m / 1024.0) * std::pow(2.0, e - 15);
                     return s ? -v : v;
                 };
                 fa = static_cast<float>(half_to_float(xa));
                 fb = static_cast<float>(half_to_float(xb));
                 const double e = static_cast<double>(fa) - fb;
                 se += e * e; sr += static_cast<double>(fa) * fa;
             }
             const double rel = sr > 0.0 ? std::sqrt(se / sr) : 0.0;
             std::cout << "XCTX2 tap_layer_" << kTapLayers[static_cast<std::size_t>(t)]
                       << " fnvA=" << std::hex << ha << " fnvB=" << hb << std::dec
                       << " diff_words=" << diff_words << "/" << host.size()
                       << " rel_l2=" << rel << "\n";
             if (ha != hb) taps_equal = false;
         }
         std::cout << "XCTX2 prefill_tap_bytes_xctx=" << (taps_equal ? "PASS" : "FAIL") << "\n";
         // X3: decode rows.
         const auto ta = greedy_step(*ctx_a);
         const auto tb = greedy_step(*ctx_b);
         std::cout << "XCTX2 token_a=" << ta << " token_b=" << tb
                   << " token_match=" << (ta == tb ? "PASS" : "FAIL") << "\n";
         const std::size_t row_bytes = static_cast<std::size_t>(5120) * sizeof(std::uint16_t);
         std::vector<std::uint16_t> ra(5120), rb(5120);
         bool rows_equal = true;
         for (int t = 0; t < kTapCount; ++t) {
             ctx_a->copy_tap_row_to_device(kTapLayers[static_cast<std::size_t>(t)], row_ptrs[t]);
             cuda_check(cudaDeviceSynchronize(), "XCTX2 row A sync");
             cuda_check(cudaMemcpy(ra.data(), row_ptrs[t], row_bytes, cudaMemcpyDeviceToHost),
                        "XCTX2 download row A");
             ctx_b->copy_tap_row_to_device(kTapLayers[static_cast<std::size_t>(t)], row_ptrs[t]);
             cuda_check(cudaDeviceSynchronize(), "XCTX2 row B sync");
             cuda_check(cudaMemcpy(rb.data(), row_ptrs[t], row_bytes, cudaMemcpyDeviceToHost),
                        "XCTX2 download row B");
             const std::uint64_t ha = fnv_bytes(ra.data(), row_bytes);
             const std::uint64_t hb = fnv_bytes(rb.data(), row_bytes);
             int dw = 0;
             double dse = 0.0, dsr = 0.0, dma = 0.0;
             for (std::size_t i = 0; i < ra.size(); ++i) {
                 if (ra[i] != rb[i]) ++dw;
                 int sa = (ra[i] >> 15) & 1, ea = (ra[i] >> 10) & 31, ma = ra[i] & 1023;
                 int sb = (rb[i] >> 15) & 1, eb = (rb[i] >> 10) & 31, mb = rb[i] & 1023;
                 double va = (ea == 0) ? (ma / 1024.0) * 0.00006103515625 : (ea == 31) ? 1e30 : (1.0 + ma / 1024.0) * std::pow(2.0, ea - 15);
                 double vb = (eb == 0) ? (mb / 1024.0) * 0.00006103515625 : (eb == 31) ? 1e30 : (1.0 + mb / 1024.0) * std::pow(2.0, eb - 15);
                 if (sa) va = -va;
                 if (sb) vb = -vb;
                 const double de = va - vb;
                 dse += de * de; dsr += va * va;
                 const double da = std::fabs(de);
                 if (da > dma) dma = da;
             }
             const double drel = dsr > 0.0 ? std::sqrt(dse / dsr) : 0.0;


             std::cout << "XCTX2 decode_layer_" << kTapLayers[static_cast<std::size_t>(t)]
                       << " fnvA=" << std::hex << ha << " fnvB=" << hb << std::dec
                       << " diff_words=" << dw << "/5120"
                       << " rel_l2=" << drel << " max_abs=" << dma << "\n";
             if (ha != hb) rows_equal = false;
         }
         std::cout << "XCTX2 decode_tap_bytes_xctx=" << (rows_equal ? "PASS" : "FAIL") << "\n";
         std::cout << "XCTX2_DONE\n";
         return 0;
     } catch (const std::exception& ex) {
         std::cerr << "XCTX2 FAILED: " << ex.what() << "\n";
         return 1;
     }
 }
