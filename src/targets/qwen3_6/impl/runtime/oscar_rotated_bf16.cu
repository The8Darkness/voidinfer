#include "targets/qwen3_6/impl/runtime/oscar_rotated_bf16.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6::oscar_internal {
namespace {

constexpr int kHeadDim = 256;

__global__ void rotate_kernel(const __nv_bfloat16* input, __nv_bfloat16* output,
                              const float* matrix, int heads, int tokens, bool transpose) {
    const std::int64_t linear = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total  = static_cast<std::int64_t>(heads) * tokens * kHeadDim;
    if (linear >= total) { return; }

    const int output_dim = static_cast<int>(linear % kHeadDim);
    const std::int64_t vector_index = linear / kHeadDim;
    const int head                  = static_cast<int>(vector_index % heads);
    const int token                 = static_cast<int>(vector_index / heads);
    const std::int64_t input_base =
        static_cast<std::int64_t>(kHeadDim) * (head + static_cast<std::int64_t>(heads) * token);
    float sum = 0.0F;
    for (int input_dim = 0; input_dim < kHeadDim; ++input_dim) {
        const float x = __bfloat162float(input[input_base + input_dim]);
        // Normal path: row vector x @ R, output[j] = sum_i x[i] R[i,j].
        // Inverse path: row vector x @ R^T, output[j] = sum_i x[i] R[j,i].
        const float r = transpose ? matrix[static_cast<std::int64_t>(output_dim) * kHeadDim +
                                          input_dim]
                                  : matrix[static_cast<std::int64_t>(input_dim) * kHeadDim +
                                           output_dim];
        sum = fmaf(x, r, sum);
    }
    output[static_cast<std::int64_t>(output_dim) + input_base] = __float2bfloat16(sum);
}

__global__ void rotate_kernel_fp32(const __nv_bfloat16* input, float* output,
                                   const float* matrix, int heads, int tokens) {
    const std::int64_t linear = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = static_cast<std::int64_t>(heads) * tokens * kHeadDim;
    if (linear >= total) { return; }

    const int output_dim = static_cast<int>(linear % kHeadDim);
    const std::int64_t vector_index = linear / kHeadDim;
    const int head = static_cast<int>(vector_index % heads);
    const int token = static_cast<int>(vector_index / heads);
    const std::int64_t input_base =
        static_cast<std::int64_t>(kHeadDim) * (head + static_cast<std::int64_t>(heads) * token);
    float sum = 0.0F;
    for (int input_dim = 0; input_dim < kHeadDim; ++input_dim) {
        const float x = __bfloat162float(input[input_base + input_dim]);
        sum = fmaf(x, matrix[static_cast<std::int64_t>(input_dim) * kHeadDim + output_dim], sum);
    }
    output[static_cast<std::int64_t>(output_dim) + input_base] = sum;
}

__global__ void reference_attention_fp32_kernel(const float* q, const float* k, const float* v,
                                                float* out, int tokens, float scale) {
    const std::int64_t linear = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = static_cast<std::int64_t>(kHeadDim) * 24 * tokens;
    if (linear >= total) { return; }

    const int output_dim = static_cast<int>(linear % kHeadDim);
    const std::int64_t vector_index = linear / kHeadDim;
    const int q_head = static_cast<int>(vector_index % 24);
    const int query_token = static_cast<int>(vector_index / 24);
    const int kv_head = q_head / 6;
    float maximum = -3.402823466e+38F;
    for (int key_token = 0; key_token <= query_token; ++key_token) {
        float dot = 0.0F;
        for (int component = 0; component < kHeadDim; ++component) {
            const std::int64_t q_index =
                component + static_cast<std::int64_t>(kHeadDim) * (q_head + 24LL * query_token);
            const std::int64_t k_index =
                component + static_cast<std::int64_t>(kHeadDim) * (kv_head + 4LL * key_token);
            dot = fmaf(q[q_index], k[k_index], dot);
        }
        maximum = fmaxf(maximum, dot * scale);
    }

    float denominator = 0.0F;
    float numerator = 0.0F;
    for (int key_token = 0; key_token <= query_token; ++key_token) {
        float dot = 0.0F;
        for (int component = 0; component < kHeadDim; ++component) {
            const std::int64_t q_index =
                component + static_cast<std::int64_t>(kHeadDim) * (q_head + 24LL * query_token);
            const std::int64_t k_index =
                component + static_cast<std::int64_t>(kHeadDim) * (kv_head + 4LL * key_token);
            dot = fmaf(q[q_index], k[k_index], dot);
        }
        const float weight = expf(dot * scale - maximum);
        denominator += weight;
        const std::int64_t v_index =
            output_dim + static_cast<std::int64_t>(kHeadDim) * (kv_head + 4LL * key_token);
        numerator = fmaf(weight, v[v_index], numerator);
    }
    out[linear] = numerator / denominator;
}

__global__ void inverse_kernel_fp32(const float* input, float* output, const float* matrix,
                                    int tokens) {
    const std::int64_t linear = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = static_cast<std::int64_t>(24) * tokens * kHeadDim;
    if (linear >= total) { return; }

    const int output_dim = static_cast<int>(linear % kHeadDim);
    const std::int64_t vector_index = linear / kHeadDim;
    const int head = static_cast<int>(vector_index % 24);
    const int token = static_cast<int>(vector_index / 24);
    const std::int64_t input_base =
        static_cast<std::int64_t>(kHeadDim) * (head + 24LL * token);
    float sum = 0.0F;
    for (int input_dim = 0; input_dim < kHeadDim; ++input_dim) {
        const float value = input[input_base + input_dim];
        sum = fmaf(value, matrix[static_cast<std::int64_t>(output_dim) * kHeadDim + input_dim],
                   sum);
    }
    output[linear] = sum;
}

__global__ void inverse_kernel_bf16_to_fp32(const __nv_bfloat16* input, float* output,
                                            const float* matrix, int tokens) {
    const std::int64_t linear = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = static_cast<std::int64_t>(24) * tokens * kHeadDim;
    if (linear >= total) { return; }

    const int output_dim = static_cast<int>(linear % kHeadDim);
    const std::int64_t vector_index = linear / kHeadDim;
    const int head = static_cast<int>(vector_index % 24);
    const int token = static_cast<int>(vector_index / 24);
    const std::int64_t input_base =
        static_cast<std::int64_t>(kHeadDim) * (head + 24LL * token);
    float sum = 0.0F;
    for (int input_dim = 0; input_dim < kHeadDim; ++input_dim) {
        const float value = __bfloat162float(input[input_base + input_dim]);
        sum = fmaf(value, matrix[static_cast<std::int64_t>(output_dim) * kHeadDim + input_dim],
                   sum);
    }
    output[linear] = sum;
}

__global__ void fp32_to_bf16_kernel(const float* input, __nv_bfloat16* output,
                                    std::int64_t elements) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) { output[index] = __float2bfloat16(input[index]); }
}

__global__ void bf16_to_fp32_kernel(const __nv_bfloat16* input, float* output,
                                    std::int64_t elements) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) { output[index] = __bfloat162float(input[index]); }
}

__global__ void matched_cache_append_kernel(const float* k, const float* v,
                                            const std::int32_t* positions, float* k_cache,
                                            float* v_cache, int tokens, int max_context) {
    const std::int64_t linear = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total  = static_cast<std::int64_t>(kHeadDim) * 4 * tokens;
    if (linear >= total) { return; }
    const int dim = static_cast<int>(linear % kHeadDim);
    const std::int64_t vector_index = linear / kHeadDim;
    const int head                  = static_cast<int>(vector_index % 4);
    const int token                 = static_cast<int>(vector_index / 4);
    const int position              = positions[token];
    if (position < 0 || position >= max_context) {
        k_cache[0] = __int_as_float(0x7fc00000);
        v_cache[0] = __int_as_float(0x7fc00000);
        return;
    }
    const std::int64_t destination = dim + static_cast<std::int64_t>(kHeadDim) *
                                               (head + 4LL * position);
    k_cache[destination] = k[linear];
    v_cache[destination] = v[linear];
}

__global__ void matched_attention_kernel(const float* q, const std::int32_t* positions,
                                         const float* k_cache, const float* v_cache, float* out,
                                         int tokens, int max_context, float scale) {
    const std::int64_t linear = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total  = static_cast<std::int64_t>(kHeadDim) * 24 * tokens;
    if (linear >= total) { return; }
    const int output_dim = static_cast<int>(linear % kHeadDim);
    const std::int64_t vector_index = linear / kHeadDim;
    const int q_head                 = static_cast<int>(vector_index % 24);
    const int query_token            = static_cast<int>(vector_index / 24);
    const int kv_head                = q_head / 6;
    const int position               = positions[query_token];
    if (position < 0 || position >= max_context) {
        out[linear] = __int_as_float(0x7fc00000);
        return;
    }

    float maximum = -3.402823466e+38F;
    for (int key_token = 0; key_token <= position; ++key_token) {
        float dot = 0.0F;
        for (int component = 0; component < kHeadDim; ++component) {
            const std::int64_t q_index =
                component + static_cast<std::int64_t>(kHeadDim) * (q_head + 24LL * query_token);
            const std::int64_t k_index =
                component + static_cast<std::int64_t>(kHeadDim) * (kv_head + 4LL * key_token);
            dot = fmaf(q[q_index], k_cache[k_index], dot);
        }
        maximum = fmaxf(maximum, dot * scale);
    }

    float denominator = 0.0F;
    float numerator   = 0.0F;
    for (int key_token = 0; key_token <= position; ++key_token) {
        float dot = 0.0F;
        for (int component = 0; component < kHeadDim; ++component) {
            const std::int64_t q_index =
                component + static_cast<std::int64_t>(kHeadDim) * (q_head + 24LL * query_token);
            const std::int64_t k_index =
                component + static_cast<std::int64_t>(kHeadDim) * (kv_head + 4LL * key_token);
            dot = fmaf(q[q_index], k_cache[k_index], dot);
        }
        const float weight = expf(dot * scale - maximum);
        denominator += weight;
        const std::int64_t v_index =
            output_dim + static_cast<std::int64_t>(kHeadDim) * (kv_head + 4LL * key_token);
        numerator = fmaf(weight, v_cache[v_index], numerator);
    }
    out[linear] = numerator / denominator;
}

void launch_rotate(const Tensor& input, Tensor& output, const float* matrix, int heads,
                   int tokens, cudaStream_t stream, bool transpose) {
    const std::int64_t total = static_cast<std::int64_t>(heads) * tokens * kHeadDim;
    constexpr int block       = 256;
    const auto grid           = static_cast<unsigned int>((total + block - 1) / block);
    rotate_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data), static_cast<__nv_bfloat16*>(output.data),
        matrix, heads, tokens, transpose);
    CUDA_CHECK(cudaGetLastError());
}

void launch_rotate_fp32(const Tensor& input, Tensor& output, const float* matrix, int heads,
                        int tokens, cudaStream_t stream) {
    const std::int64_t total = static_cast<std::int64_t>(heads) * tokens * kHeadDim;
    constexpr int block = 256;
    const auto grid = static_cast<unsigned int>((total + block - 1) / block);
    rotate_kernel_fp32<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data), static_cast<float*>(output.data), matrix,
        heads, tokens);
    CUDA_CHECK(cudaGetLastError());
}

void require_tensor(const Tensor& tensor, int heads, const char* label) {
    if (tensor.data == nullptr || tensor.dtype != DType::BF16 || !tensor.is_contiguous() ||
        tensor.ne[0] != kHeadDim || tensor.ne[1] != heads || tensor.ne[2] <= 0 ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("OSCAR rotated-BF16 ") + label +
                                    " must be contiguous BF16 [256,heads,tokens]");
    }
}

void require_fp32_tensor(const Tensor& tensor, int heads, const char* label) {
    if (tensor.data == nullptr || tensor.dtype != DType::FP32 || !tensor.is_contiguous() ||
        tensor.ne[0] != kHeadDim || tensor.ne[1] != heads || tensor.ne[2] <= 0 ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("OSCAR FP32 diagnostic ") + label +
                                    " must be contiguous FP32 [256,heads,tokens]");
    }
}

void require_positions(const Tensor& positions, std::int64_t expected, const char* label) {
    if (positions.data == nullptr || positions.dtype != DType::I32 ||
        !positions.is_contiguous() || positions.numel() != expected) {
        throw std::invalid_argument(std::string("OSCAR matched FP32 ") + label +
                                    " must be contiguous I32 with one position per token");
    }
}

} // namespace

void launch_oscar_rotate_qkv(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& q_rotated,
                             Tensor& k_rotated, Tensor& v_rotated, const float* k_matrix,
                             const float* v_matrix, cudaStream_t stream) {
    require_tensor(q, 24, "Q");
    require_tensor(k, 4, "K");
    require_tensor(v, 4, "V");
    require_tensor(q_rotated, 24, "rotated Q");
    require_tensor(k_rotated, 4, "rotated K");
    require_tensor(v_rotated, 4, "rotated V");
    if (q.ne[2] != k.ne[2] || q.ne[2] != v.ne[2]) {
        throw std::invalid_argument("OSCAR rotated-BF16 Q/K/V token counts disagree");
    }
    if (k_matrix == nullptr || v_matrix == nullptr) {
        throw std::invalid_argument("OSCAR rotated-BF16 rotation matrix is null");
    }
    launch_rotate(q, q_rotated, k_matrix, 24, q.ne[2], stream, false);
    launch_rotate(k, k_rotated, k_matrix, 4, k.ne[2], stream, false);
    launch_rotate(v, v_rotated, v_matrix, 4, v.ne[2], stream, false);
}

void launch_oscar_inverse_value(const Tensor& rotated, Tensor& recovered,
                                const float* v_matrix, cudaStream_t stream) {
    require_tensor(rotated, 24, "attention output");
    require_tensor(recovered, 24, "recovered attention output");
    if (rotated.ne[2] != recovered.ne[2]) {
        throw std::invalid_argument("OSCAR rotated-BF16 attention output token counts disagree");
    }
    if (v_matrix == nullptr) {
        throw std::invalid_argument("OSCAR rotated-BF16 inverse matrix is null");
    }
    launch_rotate(rotated, recovered, v_matrix, 24, rotated.ne[2], stream, true);
}

void launch_oscar_rotate_qkv_fp32(const Tensor& q, const Tensor& k, const Tensor& v,
                                  Tensor& q_rotated, Tensor& k_rotated, Tensor& v_rotated,
                                  const float* k_matrix, const float* v_matrix,
                                  cudaStream_t stream) {
    require_tensor(q, 24, "Q");
    require_tensor(k, 4, "K");
    require_tensor(v, 4, "V");
    require_fp32_tensor(q_rotated, 24, "rotated Q");
    require_fp32_tensor(k_rotated, 4, "rotated K");
    require_fp32_tensor(v_rotated, 4, "rotated V");
    if (q.ne[2] != k.ne[2] || q.ne[2] != v.ne[2] || q_rotated.ne[2] != q.ne[2] ||
        k_rotated.ne[2] != k.ne[2] || v_rotated.ne[2] != v.ne[2]) {
        throw std::invalid_argument("OSCAR FP32 diagnostic Q/K/V token counts disagree");
    }
    if (k_matrix == nullptr || v_matrix == nullptr) {
        throw std::invalid_argument("OSCAR FP32 diagnostic rotation matrix is null");
    }
    launch_rotate_fp32(q, q_rotated, k_matrix, 24, q.ne[2], stream);
    launch_rotate_fp32(k, k_rotated, k_matrix, 4, k.ne[2], stream);
    launch_rotate_fp32(v, v_rotated, v_matrix, 4, v.ne[2], stream);
}

void launch_oscar_reference_attention_fp32(const Tensor& q, const Tensor& k, const Tensor& v,
                                           float scale, Tensor& out, cudaStream_t stream) {
    require_fp32_tensor(q, 24, "reference Q");
    require_fp32_tensor(k, 4, "reference K");
    require_fp32_tensor(v, 4, "reference V");
    require_fp32_tensor(out, 24, "reference output");
    if (q.ne[2] != k.ne[2] || q.ne[2] != v.ne[2] || q.ne[2] != out.ne[2]) {
        throw std::invalid_argument("OSCAR FP32 diagnostic attention token counts disagree");
    }
    if (q.ne[2] > 64) {
        throw std::invalid_argument("OSCAR FP32 diagnostic attention is limited to 64 tokens");
    }
    const std::int64_t total = static_cast<std::int64_t>(kHeadDim) * 24 * q.ne[2];
    constexpr int block = 128;
    const auto grid = static_cast<unsigned int>((total + block - 1) / block);
    reference_attention_fp32_kernel<<<grid, block, 0, stream>>>(
        static_cast<const float*>(q.data), static_cast<const float*>(k.data),
        static_cast<const float*>(v.data), static_cast<float*>(out.data), q.ne[2], scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_oscar_inverse_value_fp32(const Tensor& rotated, Tensor& recovered,
                                     const float* v_matrix, cudaStream_t stream) {
    const bool input_fp32 = rotated.dtype == DType::FP32;
    if (input_fp32) {
        require_fp32_tensor(rotated, 24, "FP32 attention output");
    } else {
        require_tensor(rotated, 24, "BF16 attention output");
    }
    require_fp32_tensor(recovered, 24, "FP32 recovered attention output");
    if (rotated.ne[2] != recovered.ne[2]) {
        throw std::invalid_argument("OSCAR FP32 diagnostic inverse token counts disagree");
    }
    if (v_matrix == nullptr) {
        throw std::invalid_argument("OSCAR FP32 diagnostic inverse matrix is null");
    }
    const std::int64_t total = static_cast<std::int64_t>(24) * rotated.ne[2] * kHeadDim;
    constexpr int block = 256;
    const auto grid = static_cast<unsigned int>((total + block - 1) / block);
    if (input_fp32) {
        inverse_kernel_fp32<<<grid, block, 0, stream>>>(
            static_cast<const float*>(rotated.data), static_cast<float*>(recovered.data),
            v_matrix, rotated.ne[2]);
    } else {
        inverse_kernel_bf16_to_fp32<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(rotated.data), static_cast<float*>(recovered.data),
            v_matrix, rotated.ne[2]);
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_oscar_fp32_to_bf16(const Tensor& input, Tensor& output, cudaStream_t stream) {
    if (input.dtype != DType::FP32 || output.dtype != DType::BF16 || !input.is_contiguous() ||
        !output.is_contiguous() || input.numel() != output.numel() || input.data == nullptr ||
        output.data == nullptr) {
        throw std::invalid_argument("OSCAR FP32 diagnostic cast requires matching contiguous tensors");
    }
    constexpr int block = 256;
    const auto grid = static_cast<unsigned int>((input.numel() + block - 1) / block);
    fp32_to_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const float*>(input.data), static_cast<__nv_bfloat16*>(output.data),
        input.numel());
    CUDA_CHECK(cudaGetLastError());
}

void launch_oscar_bf16_to_fp32(const Tensor& input, Tensor& output, cudaStream_t stream) {
    if (input.dtype != DType::BF16 || output.dtype != DType::FP32 || !input.is_contiguous() ||
        !output.is_contiguous() || input.numel() != output.numel() || input.data == nullptr ||
        output.data == nullptr) {
        throw std::invalid_argument("OSCAR matched FP32 diagnostic cast requires BF16/FP32 tensors");
    }
    constexpr int block = 256;
    const auto grid = static_cast<unsigned int>((input.numel() + block - 1) / block);
    bf16_to_fp32_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data), static_cast<float*>(output.data),
        input.numel());
    CUDA_CHECK(cudaGetLastError());
}

void launch_oscar_matched_cache_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                                       float* k_cache, float* v_cache,
                                       std::uint32_t max_context, cudaStream_t stream) {
    require_fp32_tensor(k, 4, "cache K");
    require_fp32_tensor(v, 4, "cache V");
    if (k.ne[2] != v.ne[2] || k_cache == nullptr || v_cache == nullptr) {
        throw std::invalid_argument("OSCAR matched FP32 cache K/V shapes disagree");
    }
    require_positions(positions, k.ne[2], "cache positions");
    const std::int64_t total = static_cast<std::int64_t>(kHeadDim) * 4 * k.ne[2];
    constexpr int block = 256;
    const auto grid = static_cast<unsigned int>((total + block - 1) / block);
    matched_cache_append_kernel<<<grid, block, 0, stream>>>(
        static_cast<const float*>(k.data), static_cast<const float*>(v.data),
        static_cast<const std::int32_t*>(positions.data), k_cache, v_cache, k.ne[2],
        static_cast<int>(max_context));
    CUDA_CHECK(cudaGetLastError());
}

void launch_oscar_matched_attention(const Tensor& q, const Tensor& positions,
                                    const float* k_cache, const float* v_cache,
                                    std::uint32_t max_context, float scale, Tensor& out,
                                    cudaStream_t stream) {
    require_fp32_tensor(q, 24, "matched Q");
    require_fp32_tensor(out, 24, "matched output");
    if (q.ne[2] != out.ne[2] || k_cache == nullptr || v_cache == nullptr) {
        throw std::invalid_argument("OSCAR matched FP32 attention shapes disagree");
    }
    require_positions(positions, q.ne[2], "attention positions");
    const std::int64_t total = static_cast<std::int64_t>(kHeadDim) * 24 * q.ne[2];
    constexpr int block = 128;
    const auto grid = static_cast<unsigned int>((total + block - 1) / block);
    matched_attention_kernel<<<grid, block, 0, stream>>>(
        static_cast<const float*>(q.data), static_cast<const std::int32_t*>(positions.data),
        k_cache, v_cache, static_cast<float*>(out.data), q.ne[2], static_cast<int>(max_context),
        scale);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::qwen3_6::oscar_internal
