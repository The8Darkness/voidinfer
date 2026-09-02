#pragma once

// Official FutureMLS-Lab OSCAR INT2 reference-direction codec.
//
// This is intentionally separate from oscar_codec.cuh.  The latter is the legacy
// runtime Q2/VeriCache path (different clipping, metadata, and layouts).  This
// host implementation is used by the D2.1 parity gate only; it is not wired into
// live attention or mixed-precision cache management.

#include <array>
#include <cstdint>

namespace ninfer::ops {

inline constexpr int kOscarInt2G128HeadDim       = 256;
inline constexpr int kOscarInt2G128GroupSize     = 128;
inline constexpr int kOscarInt2G128Groups        = 2;
inline constexpr int kOscarInt2G128Bits          = 2;
inline constexpr int kOscarInt2G128Levels        = 3;
inline constexpr int kOscarInt2G128CodeBytes     = 64;
inline constexpr int kOscarInt2G128MetadataItems = 4; // [scale, zero] per group

struct OscarInt2G128EncodedRow {
    // These fields are exposed so the parity test can audit every numerical boundary.
    std::array<float, kOscarInt2G128HeadDim> clipped{};
    std::array<float, kOscarInt2G128MetadataItems> scales_zeros{};
    std::array<std::uint8_t, kOscarInt2G128HeadDim> symbols{};
    std::array<std::uint8_t, kOscarInt2G128CodeBytes> packed{};
};

// Encode one finite D=256 row using the official OSCAR clipped groupwise INT2
// equations. clip_ratio <= 0 disables the official percentile clip; otherwise
// the threshold is sorted_abs[int(clip_ratio * D)] with index clamped to D-1.
OscarInt2G128EncodedRow oscar_int2_g128_encode(const float* values, int count,
                                               float clip_ratio);

// Decode one encoded row.  The output is FP32 and uses the official
// (symbol - zero_point) * scale reconstruction.
void oscar_int2_g128_decode(const OscarInt2G128EncodedRow& encoded, float* output, int count);

} // namespace ninfer::ops
