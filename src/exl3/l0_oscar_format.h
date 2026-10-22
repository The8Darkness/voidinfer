#pragma once
// L0 OSCAR code format (l0_oscar.cuh), shared by the device kernels and the
// host-side parking / pinned pools. Per committed row and KV head: 256 rotated
// dims as asymmetric codes and FP16 (scale, zero) per 128-dim group.
//   INT4 (default): 4-bit codes over the full range, two dims per byte (dim 2b
//                   in the low nibble of byte b); 17.4 KiB per token.
//   INT2:           2-bit codes clipped at the 0.96 (K) / 0.92 (V) per-row
//                   quantile, four dims per byte (dim 4b + i in bits 2i);
//                   9.2 KiB per token, about 4x the KV error of INT4.
// VOIDINFER_EXL3_L0_CODE_BITS (4 or 2) selects the width for the process; the
// kernels are instantiated per width (CodeFormat<Bits>).
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace voidinfer::exl3::l0_oscar_format {

template <int Bits> struct CodeFormat {
    static_assert(Bits == 2 || Bits == 4, "L0 OSCAR codes are INT2 or INT4");
    static constexpr int bits = Bits;
    static constexpr int levels = (1 << Bits) - 1;
    static constexpr int code_bytes = 256 * Bits / 8;   // per row and KV head
};
constexpr int kMetaHalves = 4;                     // scale0, zero0, scale1, zero1
constexpr int kMaxCodeBytes = CodeFormat<4>::code_bytes;
// One row of all 4 KV heads, K and V: FP16 meta (codes: row_code_bytes()).
constexpr std::size_t kRowMetaBytes = static_cast<std::size_t>(2) * 4 * kMetaHalves * 2;

inline int code_bits() {
    static const int value = [] {
        const char* text = std::getenv("VOIDINFER_EXL3_L0_CODE_BITS");
        if (!text || !*text || std::strcmp(text, "4") == 0) return 4;
        if (std::strcmp(text, "2") == 0) return 2;
        throw std::invalid_argument("VOIDINFER_EXL3_L0_CODE_BITS must be 4 or 2");
    }();
    return value;
}
inline int code_bytes() { return 256 * code_bits() / 8; }
// One row of all 4 KV heads, K and V codes.
inline std::size_t row_code_bytes() { return static_cast<std::size_t>(2) * 4 * code_bytes(); }

} // namespace voidinfer::exl3::l0_oscar_format
