#include "ninfer/ops/dflash2_qkv_proj.h"

#include "ops/launcher/dflash2_qkv_proj.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t columns,
                    const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != columns ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("dflash2_qkv_proj: invalid ") + label);
    }
}

} // namespace

void dflash2_qkv_proj(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                     cudaStream_t stream) {
    constexpr std::int32_t kInputRows = 5120;
    constexpr std::int32_t kQueryRows = 4096;
    constexpr std::int32_t kKvRows = 1024;
    constexpr std::int32_t kParentRows = kQueryRows + 2 * kKvRows;
    const std::int32_t columns = x.ne[1];
    if (columns <= 0) {
        throw std::invalid_argument("dflash2_qkv_proj: T must be positive");
    }
    require_matrix(x, kInputRows, columns, "x");
    require_matrix(q, kQueryRows, columns, "q");
    require_matrix(k, kKvRows, columns, "k");
    require_matrix(v, kKvRows, columns, "v");
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != kParentRows || weight.k != kInputRows ||
        weight.shape[0] != kParentRows || weight.shape[1] != kInputRows ||
        weight.padded_shape[0] != kParentRows || weight.padded_shape[1] != kInputRows ||
        weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument("dflash2_qkv_proj: invalid W8G32_F16S parent");
    }
    if (q.data == k.data || q.data == v.data || k.data == v.data || x.data == q.data ||
        x.data == k.data || x.data == v.data) {
        throw std::invalid_argument("dflash2_qkv_proj: tensors must not alias");
    }
    detail::dflash2_qkv_proj_launch(x, weight, q, k, v, stream);
}

} // namespace ninfer::ops
