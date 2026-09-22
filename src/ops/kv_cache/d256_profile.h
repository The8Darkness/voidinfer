#pragma once

#include "core/dtype.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

inline constexpr std::int32_t kD256KVCacheHeadDim = 256;
inline constexpr std::int32_t kD256OscarQuantGroup = 128;
inline constexpr std::int32_t kD256OscarCodeExtent = 64;
inline constexpr std::int32_t kD256OscarScaleExtent = 2;

struct D256KVCacheProfile {
    DType code_dtype;
    DType scale_dtype;
    std::int32_t quant_group;
    std::int32_t scale_leading_extent;
    std::int32_t code_leading_extent;
};

inline D256KVCacheProfile d256_kv_cache_profile(DType dtype) {
    switch (dtype) {
    case DType::BF16:
        return {DType::BF16, DType::BF16, 0, 0, kD256KVCacheHeadDim};
    case DType::I8:
        return {DType::I8, DType::FP16, 64, 4, kD256KVCacheHeadDim};
    case DType::FP8_E4M3FN:
        return {DType::FP8_E4M3FN, DType::FP16, 256, 1, kD256KVCacheHeadDim};
    case DType::U8:
        // DType::U8 is the storage type for packed NVFP4 code bytes. The public enum selects
        // this profile; ordinary U8 tensors are not accepted by the KV append/attention APIs.
        return {DType::U8, DType::FP8_E4M3FN, 16, kD256KVCacheHeadDim / 16,
                kD256KVCacheHeadDim / 2};
    default:
        throw std::invalid_argument("unsupported D256 KV-cache dtype");
    }
}

// DType::U8 is shared by the legacy NVFP4 and OSCAR packed routes.  The quantization group is
// part of the public paged-cache view, so keep the profiles explicit instead of inferring OSCAR
// from the storage dtype alone.
inline D256KVCacheProfile d256_kv_cache_profile(DType dtype, std::int32_t quant_group) {
    if (dtype == DType::U8 && quant_group == kD256OscarQuantGroup) {
        return {DType::U8, DType::BF16, kD256OscarQuantGroup, kD256OscarScaleExtent,
                kD256OscarCodeExtent};
    }
    return d256_kv_cache_profile(dtype);
}

} // namespace ninfer::ops
