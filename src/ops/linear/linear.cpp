#include "ninfer/ops/linear.h"

#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/w8/w8_dispatch.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t checked_numel(const Tensor& tensor, const char* label) {
    std::int64_t total = 1;
    for (const std::int32_t extent : tensor.ne) {
        if (extent <= 0) {
            throw std::invalid_argument(std::string("linear: ") + label +
                                        " dimensions must be positive");
        }
        if (total > std::numeric_limits<std::int64_t>::max() / extent) {
            throw std::overflow_error("linear: tensor size overflows int64");
        }
        total *= extent;
    }
    return total;
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_linear_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear: invalid compute policy");
}

void validate_linear_semantics(const Tensor& x, const Weight& w, const Tensor& out,
                               LinearPolicy policy) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear: x/out must be BF16");
    }
    (void)checked_numel(x, "x");
    (void)checked_numel(out, "out");
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("linear: x must have shape [K,T]");
    }
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("linear: out must have shape [N,T]");
    }
    if (w.n <= 0 || w.k <= 0) {
        throw std::invalid_argument("linear: weight n/k must be positive");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("linear: expected [K,T] x [N,K] -> [N,T]");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear: x/out must be non-null and 16-byte aligned");
    }
    validate_linear_policy(policy);
}

void dispatch_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream) {
    switch (w.qtype) {
    case QType::Q4G64_F16S:
        detail::q4_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q5G64_F16S:
        detail::q5_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q6G64_F16S:
        detail::q6_dispatch(x, w, out, policy, stream);
        return;
    case QType::W8G32_F16S:
        detail::w8_dispatch(x, w, out, policy, stream);
        return;
    case QType::BF16_CTRL:
        detail::bf16_dispatch(x, w, out, policy, stream);
        return;
    case QType::NVFP4:
        detail::nvfp4_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP8_E4M3FN_ROW_BF16S:
        detail::fp8_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP32_CTRL:
    case QType::I32_CTRL:
        break;
    }
    throw std::invalid_argument("linear: unsupported weight qtype");
}

} // namespace

std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                            std::int32_t input_rows, LinearPolicy policy,
                                            std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_linear_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear workspace: invalid token interval");
    }

    switch (qtype) {
    case QType::Q4G64_F16S:
        (void)detail::select_q4_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q4_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q5G64_F16S:
        (void)detail::select_q5_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q5_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q6G64_F16S:
        (void)detail::select_q6_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q6_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::W8G32_F16S:
        (void)detail::select_w8_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_w8_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::BF16_CTRL:
        (void)detail::select_bf16_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_bf16_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::NVFP4:
        if (!detail::is_nvfp4_linear_problem(output_rows, input_rows) ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
            throw std::invalid_argument("linear workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                             min_tokens, max_tokens);
    case QType::FP8_E4M3FN_ROW_BF16S:
        return detail::fp8_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                           min_tokens, max_tokens);
    case QType::FP32_CTRL:
    case QType::I32_CTRL:
        break;
    }
    throw std::invalid_argument("linear workspace: unsupported weight qtype");
}

void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, policy);
    dispatch_linear(x, w, out, policy, &workspace, stream);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, LinearPolicy::A16Only);
    dispatch_linear(x, w, out, LinearPolicy::A16Only, nullptr, stream);
}

void linear_argmax(const Tensor& x, const Weight& w, Tensor& out, std::int32_t valid_rows,
                   Tensor& scratch, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || out.dtype != DType::I32) {
        throw std::invalid_argument("linear_argmax: x must be BF16 and out must be I32");
    }
    if (!x.is_contiguous() || !out.is_contiguous() || !scratch.is_contiguous() ||
        x.data == nullptr || out.data == nullptr || scratch.data == nullptr) {
        throw std::invalid_argument("linear_argmax: x/out/scratch must be contiguous and non-null");
    }
    if (x.ne[2] != 1 || x.ne[3] != 1 || out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1 ||
        x.ne[0] != 5120 || out.ne[0] != x.ne[1] || x.ne[1] < 1 || x.ne[1] > 48) {
        throw std::invalid_argument("linear_argmax: expected BF16 [5120,T] -> I32 [T], 1<=T<=48");
    }
    if (w.qtype != QType::FP8_E4M3FN_ROW_BF16S || w.n != 248320 || w.k != 5120) {
        throw std::invalid_argument("linear_argmax: unsupported vocabulary weight");
    }
    if (valid_rows <= 0 || valid_rows > w.n) {
        throw std::invalid_argument("linear_argmax: valid_rows is outside the vocabulary domain");
    }
    if (scratch.dtype != DType::BF16) {
        throw std::invalid_argument("linear_argmax: scratch must be BF16");
    }
    constexpr std::size_t kPartialRows = 248320U / 16U;
    const std::size_t partial_count = kPartialRows * static_cast<std::size_t>(x.ne[1]);
    const std::size_t index_offset =
        (partial_count * sizeof(std::uint16_t) + 15U) & ~std::size_t(15U);
    const std::size_t required_bytes =
        index_offset + partial_count * sizeof(std::int32_t);
    if (scratch.bytes() < required_bytes) {
        throw std::invalid_argument("linear_argmax: scratch is too small for row-tile winners");
    }
    if ((reinterpret_cast<std::uintptr_t>(x.data) & 15U) != 0U) {
        throw std::invalid_argument("linear_argmax: x must be 16-byte aligned");
    }
    detail::launch_fp8_vocabulary_a16_small_t_argmax(x, w, out, valid_rows, scratch, stream);
}

} // namespace ninfer::ops
