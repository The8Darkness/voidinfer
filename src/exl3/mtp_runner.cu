#include "exl3/mtp_runner.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::exl3 {
namespace {

constexpr int kHidden = static_cast<int>(kExl3MtpHiddenWidth);
constexpr int kQHeads = 24;
constexpr int kKVHeads = 4;
constexpr int kHeadDim = 256;
constexpr int kQSize = kQHeads * kHeadDim;
constexpr int kKVSize = kKVHeads * kHeadDim;
constexpr int kIntermediate = 17'408;
constexpr int kVocab = static_cast<int>(kExl3MtpVocabSize);
constexpr int kTokenDomain = static_cast<int>(kExl3MtpTokenDomain);
constexpr int kRopeDim = 64;
constexpr float kRopeTheta = 10'000'000.0f;
constexpr float kRmsEps = 1.0e-6f;

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

struct DeviceBuffer {
    void* data = nullptr;
    std::size_t bytes = 0;

    explicit DeviceBuffer(std::size_t extent, const char* label) : bytes(extent) {
        check(cudaMalloc(&data, bytes), label);
    }
    ~DeviceBuffer() {
        if (data) (void)cudaFree(data);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

const Exl3MtpDeviceTensor* find_tensor(const Exl3MtpExecutionBinding& binding,
                                       std::string_view name) {
    for (const auto& tensor : binding.tensors)
        if (tensor.name == name) return &tensor;
    return nullptr;
}

const std::uint16_t* u16_tensor(const Exl3MtpExecutionBinding& binding,
                                std::string_view name, std::size_t bytes) {
    const auto* tensor = find_tensor(binding, name);
    if (!tensor || tensor->data == 0 || tensor->bytes != bytes)
        throw std::invalid_argument("native MTP tensor binding extent mismatch");
    return reinterpret_cast<const std::uint16_t*>(tensor->data);
}

const std::int32_t* i32_tensor(const Exl3MtpExecutionBinding& binding,
                               std::string_view name) {
    const auto* tensor = find_tensor(binding, name);
    if (!tensor || tensor->data == 0 || tensor->bytes != sizeof(std::int32_t))
        throw std::invalid_argument("native MTP multiplier binding extent mismatch");
    return reinterpret_cast<const std::int32_t*>(tensor->data);
}

Exl3CudaLinearWeights linear_weights(const Exl3MtpExecutionBinding& binding,
                                     std::string_view prefix, int in_features,
                                     int out_features) {
    const std::string base(prefix);
    return {
        u16_tensor(binding, base + ".trellis",
                   static_cast<std::size_t>(in_features / 16) *
                       static_cast<std::size_t>(out_features / 16) * 64U * sizeof(std::uint16_t)),
        u16_tensor(binding, base + ".suh",
                   static_cast<std::size_t>(in_features) * sizeof(std::uint16_t)),
        u16_tensor(binding, base + ".svh",
                   static_cast<std::size_t>(out_features) * sizeof(std::uint16_t)),
        i32_tensor(binding, base + ".mul1")};
}

Exl3CudaLinearMetadata linear_metadata(int in_features, int out_features) {
    return {.in_features = in_features,
            .out_features = out_features,
            .K = 4,
            .mcg = false,
            .mul1 = true,
            .has_bias = false};
}

__device__ float bf16_to_float_device(std::uint16_t bits) {
    union Value {
        std::uint32_t u;
        float f;
    } value{static_cast<std::uint32_t>(bits) << 16U};
    return value.f;
}

__device__ std::uint16_t half_bits(float value) {
    return __half_as_ushort(__float2half_rn(value));
}

__global__ void embed_one_kernel(const std::uint16_t* embedding, const std::int32_t* token,
                                 std::uint16_t* output) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < kHidden) {
        const auto id = token[0];
        output[i] = half_bits(bf16_to_float_device(embedding[static_cast<std::size_t>(id) * kHidden + i]));
    }
}

__global__ void rmsnorm_bf16_weight_kernel(const std::uint16_t* input,
                                           const std::uint16_t* weight,
                                           std::uint16_t* output, int features) {
    __shared__ float partial[512];
    const int lane = static_cast<int>(threadIdx.x);
    float sum = 0.0f;
    for (int i = lane; i < features; i += blockDim.x) {
        const float value = __half2float(__ushort_as_half(input[i]));
        sum += value * value;
    }
    partial[lane] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / static_cast<float>(features) + kRmsEps);
    for (int i = lane; i < features; i += blockDim.x) {
        const float value = __half2float(__ushort_as_half(input[i])) * inv;
        // Native target MTP RMSNorm uses the unit-offset epilogue: the
        // stored BF16 scale is applied as (weight + 1), not as weight.
        output[i] = half_bits(value * (bf16_to_float_device(weight[i]) + 1.0f));
    }
}

__global__ void concat_fc_input_kernel(const std::uint16_t* embedding,
                                       const std::uint16_t* hidden,
                                       std::uint16_t* output) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < kHidden) {
        output[i] = embedding[i];
        output[kHidden + i] = hidden[i];
    }
}

__global__ void split_q_gate_kernel(const std::uint16_t* q_gate,
                                    std::uint16_t* q, std::uint16_t* gate) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < kQSize) {
        q[i] = q_gate[i];
        gate[i] = q_gate[kQSize + i];
    }
}

__global__ void rmsnorm_heads_bf16_kernel(const std::uint16_t* input,
                                          const std::uint16_t* weight,
                                          std::uint16_t* output, int heads) {
    const int head = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (head >= heads) return;
    __shared__ float partial[256];
    float sum = 0.0f;
    for (int i = lane; i < kHeadDim; i += blockDim.x) {
        const float value = __half2float(__ushort_as_half(input[head * kHeadDim + i]));
        sum += value * value;
    }
    partial[lane] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / static_cast<float>(kHeadDim) + kRmsEps);
    for (int i = lane; i < kHeadDim; i += blockDim.x) {
        const float value = __half2float(__ushort_as_half(input[head * kHeadDim + i]));
        // The per-head MTP RMSNorms use the same unit-offset epilogue.
        output[head * kHeadDim + i] = half_bits(
            value * inv * (bf16_to_float_device(weight[i]) + 1.0f));
    }
}

__global__ void rope_one_kernel(std::uint16_t* q, std::uint16_t* k, int position) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < kQSize) {
        const int channel = i % kHeadDim;
        const int head = i / kHeadDim;
        // One thread owns each rotary pair. This permits in-place output
        // without a read/write race and matches the native first-half/second-
        // half pairing (0<->32 ... 31<->63).
        if (channel < kRopeDim / 2) {
            const int pair = channel;
            const float angle = static_cast<float>(position) *
                                powf(kRopeTheta, -2.0f * static_cast<float>(pair) /
                                                       static_cast<float>(kRopeDim));
            const float c = cosf(angle), s = sinf(angle);
            const float x0 = __half2float(__ushort_as_half(q[i]));
            const float x1 = __half2float(__ushort_as_half(q[i + kRopeDim / 2]));
            q[i] = half_bits(x0 * c - x1 * s);
            q[i + kRopeDim / 2] = half_bits(x0 * s + x1 * c);
            if (head < kKVHeads) {
                const int ki = head * kHeadDim + channel;
                const float k0 = __half2float(__ushort_as_half(k[ki]));
                const float k1 = __half2float(__ushort_as_half(k[ki + kRopeDim / 2]));
                k[ki] = half_bits(k0 * c - k1 * s);
                k[ki + kRopeDim / 2] = half_bits(k0 * s + k1 * c);
            }
        }
    }
}

__global__ void append_kv_kernel(const std::uint16_t* key, const std::uint16_t* value,
                                 std::uint16_t* key_cache, std::uint16_t* value_cache,
                                 std::uint32_t position) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < kKVSize) {
        key_cache[static_cast<std::size_t>(position) * kKVSize + i] = key[i];
        value_cache[static_cast<std::size_t>(position) * kKVSize + i] = value[i];
    }
}

// Exact chronological one-layer attention for the small bridge surface. The
// score/value accumulation order is scalar and deterministic. This kernel is
// intentionally separate from the production HostKV verifier operators.
__global__ void prefix_attention_kernel(const std::uint16_t* query,
                                        const std::uint16_t* gate,
                                        const std::uint16_t* key_cache,
                                        const std::uint16_t* value_cache,
                                        float* scores, std::uint16_t* output,
                                        std::uint32_t visible_rows) {
    const int head = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (head >= kQHeads) return;
    __shared__ float maximum;
    __shared__ float denominator;
    if (lane == 0) {
        float max_value = -3.402823466e+38F;
        const int kv_head = head / (kQHeads / kKVHeads);
        for (std::uint32_t row = 0; row < visible_rows; ++row) {
            float dot = 0.0f;
            const auto* q = query + head * kHeadDim;
            const auto* k = key_cache + (static_cast<std::size_t>(row) * kKVHeads + kv_head) * kHeadDim;
            for (int d = 0; d < kHeadDim; ++d)
                dot += __half2float(__ushort_as_half(q[d])) *
                       __half2float(__ushort_as_half(k[d]));
            const float score = dot * 0.0625f;
            scores[static_cast<std::size_t>(head) * visible_rows + row] = score;
            max_value = fmaxf(max_value, score);
        }
        maximum = max_value;
    }
    __syncthreads();
    if (lane == 0) {
        float total = 0.0f;
        for (std::uint32_t row = 0; row < visible_rows; ++row) {
            auto& score = scores[static_cast<std::size_t>(head) * visible_rows + row];
            score = expf(score - maximum);
            total += score;
        }
        denominator = total;
    }
    __syncthreads();
    for (int d = lane; d < kHeadDim; d += blockDim.x) {
        float value = 0.0f;
        const int kv_head = head / (kQHeads / kKVHeads);
        for (std::uint32_t row = 0; row < visible_rows; ++row) {
            const auto* v = value_cache + (static_cast<std::size_t>(row) * kKVHeads + kv_head) * kHeadDim;
            value += scores[static_cast<std::size_t>(head) * visible_rows + row] /
                     denominator * __half2float(__ushort_as_half(v[d]));
        }
        const float g = __half2float(__ushort_as_half(gate[head * kHeadDim + d]));
        output[head * kHeadDim + d] = half_bits(1.0f / (1.0f + expf(-g)) * value);
    }
}

__global__ void residual_kernel(const std::uint16_t* input, const std::uint16_t* residual,
                                std::uint16_t* output, int count) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        output[i] = half_bits(__half2float(__ushort_as_half(input[i])) +
                              __half2float(__ushort_as_half(residual[i])));
}

__global__ void silu_mul_kernel(const std::uint16_t* gate, const std::uint16_t* up,
                                std::uint16_t* output) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < kIntermediate) {
        const float g = __half2float(__ushort_as_half(gate[i]));
        const float u = __half2float(__ushort_as_half(up[i]));
        output[i] = half_bits((g / (1.0f + expf(-g))) * u);
    }
}

__global__ void argmax_kernel(const std::uint16_t* logits, std::int32_t* token) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float best = -3.402823466e+38F;
        int best_id = 0;
        for (int i = 0; i < kTokenDomain; ++i) {
            const float value = __half2float(__ushort_as_half(logits[i]));
            if (value > best) {
                best = value;
                best_id = i;
            }
        }
        token[0] = best_id;
    }
}

} // namespace

struct Exl3NativeMtpOneStepExecutor::Impl {
    Exl3MtpBinding manifest;
    std::vector<Exl3MtpDeviceTensor> tensors;
    Exl3MtpExecutionBinding binding;
    bool admitted = false;

    std::array<std::unique_ptr<Exl3CudaLinearWorkspace>, 9> projections;
    std::vector<std::unique_ptr<DeviceBuffer>> buffers;
    std::uint16_t* embedding = nullptr;
    std::uint16_t* hidden_norm = nullptr;
    std::uint16_t* fc_input = nullptr;
    std::uint16_t* fc_output = nullptr;
    std::uint16_t* attention_input = nullptr;
    std::uint16_t* q_gate = nullptr;
    std::uint16_t* q = nullptr;
    std::uint16_t* gate = nullptr;
    std::uint16_t* key = nullptr;
    std::uint16_t* value = nullptr;
    std::uint16_t* query_norm = nullptr;
    std::uint16_t* key_norm = nullptr;
    std::uint16_t* attention_output = nullptr;
    std::uint16_t* output_residual = nullptr;
    std::uint16_t* post_norm = nullptr;
    std::uint16_t* gate_projection = nullptr;
    std::uint16_t* up_projection = nullptr;
    std::uint16_t* activated = nullptr;
    std::int32_t* token = nullptr;
    float* scores = nullptr;

    explicit Impl(Exl3MtpExecutionBinding source) : binding(source) {
        if (!source.valid()) return;
        manifest = *source.manifest;
        tensors.assign(source.tensors.begin(), source.tensors.end());
        binding.manifest = &manifest;
        binding.tensors = tensors;
        if (binding.shared_target.lm_head_weights.trellis == nullptr ||
            binding.shared_target.lm_head_weights.suh == nullptr ||
            binding.shared_target.lm_head_weights.svh == nullptr ||
            binding.shared_target.lm_head_weights.mul1 == nullptr ||
            binding.shared_target.lm_head_metadata.K != 6 ||
            binding.shared_target.lm_head_metadata.in_features != kHidden ||
            binding.shared_target.lm_head_metadata.out_features != kVocab)
            return;

        auto make_buffer = [&](std::size_t bytes, const char* label) -> std::uint16_t* {
            buffers.push_back(std::make_unique<DeviceBuffer>(bytes, label));
            return static_cast<std::uint16_t*>(buffers.back()->data);
        };
        embedding = make_buffer(kHidden * sizeof(std::uint16_t), "allocate native MTP embedding");
        hidden_norm = make_buffer(kHidden * sizeof(std::uint16_t), "allocate native MTP hidden norm");
        fc_input = make_buffer(2U * kHidden * sizeof(std::uint16_t), "allocate native MTP FC input");
        fc_output = make_buffer(kHidden * sizeof(std::uint16_t), "allocate native MTP FC output");
        attention_input = make_buffer(kHidden * sizeof(std::uint16_t), "allocate native MTP attention input");
        q_gate = make_buffer(2U * kQSize * sizeof(std::uint16_t), "allocate native MTP Q/G");
        q = make_buffer(kQSize * sizeof(std::uint16_t), "allocate native MTP Q");
        gate = make_buffer(kQSize * sizeof(std::uint16_t), "allocate native MTP gate");
        key = make_buffer(kKVSize * sizeof(std::uint16_t), "allocate native MTP K");
        value = make_buffer(kKVSize * sizeof(std::uint16_t), "allocate native MTP V");
        query_norm = make_buffer(kQSize * sizeof(std::uint16_t), "allocate native MTP Q norm");
        key_norm = make_buffer(kKVSize * sizeof(std::uint16_t), "allocate native MTP K norm");
        attention_output = make_buffer(kQSize * sizeof(std::uint16_t), "allocate native MTP attention output");
        output_residual = make_buffer(kHidden * sizeof(std::uint16_t), "allocate native MTP residual");
        post_norm = make_buffer(kHidden * sizeof(std::uint16_t), "allocate native MTP post norm");
        gate_projection = make_buffer(kIntermediate * sizeof(std::uint16_t), "allocate native MTP gate projection");
        up_projection = make_buffer(kIntermediate * sizeof(std::uint16_t), "allocate native MTP up projection");
        activated = make_buffer(kIntermediate * sizeof(std::uint16_t), "allocate native MTP activation");
        token = reinterpret_cast<std::int32_t*>(make_buffer(sizeof(std::int32_t), "allocate native MTP token"));

        if (!binding.mtp_kv.valid()) return;
        const std::uint32_t kv_capacity = binding.mtp_kv.capacity;
        scores = reinterpret_cast<float*>(make_buffer(
            std::max<std::size_t>(1U, static_cast<std::size_t>(kv_capacity) * kQHeads) *
                sizeof(float), "allocate native MTP attention scores"));

        projections[0] = std::make_unique<Exl3CudaLinearWorkspace>(kHidden * 2, kHidden, 1);
        projections[1] = std::make_unique<Exl3CudaLinearWorkspace>(kHidden, 2 * kQSize, 1);
        projections[2] = std::make_unique<Exl3CudaLinearWorkspace>(kHidden, kKVSize, 1);
        projections[3] = std::make_unique<Exl3CudaLinearWorkspace>(kHidden, kKVSize, 1);
        projections[4] = std::make_unique<Exl3CudaLinearWorkspace>(kQSize, kHidden, 1);
        projections[5] = std::make_unique<Exl3CudaLinearWorkspace>(kHidden, kIntermediate, 1);
        projections[6] = std::make_unique<Exl3CudaLinearWorkspace>(kHidden, kIntermediate, 1);
        projections[7] = std::make_unique<Exl3CudaLinearWorkspace>(kIntermediate, kHidden, 1);
        projections[8] = std::make_unique<Exl3CudaLinearWorkspace>(kHidden, kVocab, 1);
        admitted = true;
    }

    const std::uint16_t* norm(std::string_view name, int features = kHidden) const {
        return u16_tensor(binding, name,
                          static_cast<std::size_t>(features) * sizeof(std::uint16_t));
    }

    bool enqueue(const Exl3MtpStepInput& input, const Exl3MtpStepOutput& output,
                 std::uintptr_t retained_stream) {
        if (!admitted || !binding.mtp_kv.valid() || !input.valid() ||
            input.model_identity != binding.model_identity ||
            retained_stream == 0 ||
            !output.hidden.exact(kHidden, 1, static_cast<std::size_t>(kHidden) * 2U) ||
            !output.logits.exact(kVocab, 1, static_cast<std::size_t>(kVocab) * 2U) ||
            !output.predicted_token.exact(1, 1, sizeof(std::int32_t))) return false;
        const auto stream = reinterpret_cast<cudaStream_t>(retained_stream);
        if (input.position != static_cast<std::int64_t>(binding.mtp_kv.valid_rows) ||
            input.position >= static_cast<std::int64_t>(binding.mtp_kv.capacity))
            return false;

        check(cudaMemcpyAsync(token, &input.token, sizeof(input.token),
                              cudaMemcpyHostToDevice, stream),
              "upload native MTP input token");
        const auto* embedding_table = reinterpret_cast<const std::uint16_t*>(
            binding.shared_target.token_embedding.data);
        embed_one_kernel<<<(kHidden + 255) / 256, 256, 0, stream>>>(embedding_table, token, embedding);
        check(cudaGetLastError(), "launch native MTP embedding");
        rmsnorm_bf16_weight_kernel<<<1, 512, 0, stream>>>(
            embedding, norm("mtp.pre_fc_norm_embedding.weight"), embedding, kHidden);
        check(cudaGetLastError(), "launch native MTP embedding norm");
        rmsnorm_bf16_weight_kernel<<<1, 512, 0, stream>>>(
            reinterpret_cast<const std::uint16_t*>(input.hidden.data),
            norm("mtp.pre_fc_norm_hidden.weight"), hidden_norm, kHidden);
        check(cudaGetLastError(), "launch native MTP hidden norm");
        concat_fc_input_kernel<<<(kHidden + 255) / 256, 256, 0, stream>>>(
            embedding, hidden_norm, fc_input);
        check(cudaGetLastError(), "launch native MTP FC input pack");

        projections[0]->forward(linear_weights(binding, "mtp.fc", 2 * kHidden, kHidden),
                                linear_metadata(2 * kHidden, kHidden), fc_input, fc_output, 1,
                                stream, Exl3CudaLinearAdmission::native_mtp_one_step);
        rmsnorm_bf16_weight_kernel<<<1, 512, 0, stream>>>(
            fc_output, norm("mtp.layers.0.input_layernorm.weight"), attention_input, kHidden);
        check(cudaGetLastError(), "launch native MTP input norm");
        projections[1]->forward(linear_weights(binding, "mtp.layers.0.self_attn.q_proj",
                                               kHidden, 2 * kQSize),
                                linear_metadata(kHidden, 2 * kQSize), attention_input, q_gate, 1,
                                stream, Exl3CudaLinearAdmission::native_mtp_one_step);
        projections[2]->forward(linear_weights(binding, "mtp.layers.0.self_attn.k_proj",
                                               kHidden, kKVSize),
                                linear_metadata(kHidden, kKVSize), attention_input, key, 1,
                                stream, Exl3CudaLinearAdmission::native_mtp_one_step);
        projections[3]->forward(linear_weights(binding, "mtp.layers.0.self_attn.v_proj",
                                               kHidden, kKVSize),
                                linear_metadata(kHidden, kKVSize), attention_input, value, 1,
                                stream, Exl3CudaLinearAdmission::native_mtp_one_step);
        split_q_gate_kernel<<<(kQSize + 255) / 256, 256, 0, stream>>>(q_gate, q, gate);
        check(cudaGetLastError(), "launch native MTP Q/G split");
        rmsnorm_heads_bf16_kernel<<<kQHeads, kHeadDim, 0, stream>>>(
            q, norm("mtp.layers.0.self_attn.q_norm.weight", kHeadDim), query_norm, kQHeads);
        rmsnorm_heads_bf16_kernel<<<kKVHeads, kHeadDim, 0, stream>>>(
            key, norm("mtp.layers.0.self_attn.k_norm.weight", kHeadDim), key_norm, kKVHeads);
        rope_one_kernel<<<(kQSize + 255) / 256, 256, 0, stream>>>(
            query_norm, key_norm, static_cast<int>(input.position));
        check(cudaGetLastError(), "launch native MTP RoPE");

        append_kv_kernel<<<(kKVSize + 255) / 256, 256, 0, stream>>>(
            key_norm, value, reinterpret_cast<std::uint16_t*>(binding.mtp_kv.key.data),
            reinterpret_cast<std::uint16_t*>(binding.mtp_kv.value.data),
            static_cast<std::uint32_t>(input.position));
        check(cudaGetLastError(), "append native MTP KV row");
        prefix_attention_kernel<<<kQHeads, kHeadDim, 0, stream>>>(
            query_norm, gate,
            reinterpret_cast<const std::uint16_t*>(binding.mtp_kv.key.data),
            reinterpret_cast<const std::uint16_t*>(binding.mtp_kv.value.data), scores,
            attention_output, static_cast<std::uint32_t>(input.position + 1));
        check(cudaGetLastError(), "launch native MTP prefix attention");
        const std::uint16_t* attention = attention_output;
        projections[4]->forward(linear_weights(binding, "mtp.layers.0.self_attn.o_proj",
                                               kQSize, kHidden),
                                linear_metadata(kQSize, kHidden), attention, output_residual, 1,
                                stream, Exl3CudaLinearAdmission::native_mtp_one_step);
        residual_kernel<<<(kHidden + 255) / 256, 256, 0, stream>>>(
            output_residual, fc_output, attention_input, kHidden);
        check(cudaGetLastError(), "launch native MTP attention residual");
        rmsnorm_bf16_weight_kernel<<<1, 512, 0, stream>>>(
            attention_input, norm("mtp.layers.0.post_attention_layernorm.weight"), post_norm,
            kHidden);
        check(cudaGetLastError(), "launch native MTP post-attention norm");
        projections[5]->forward(linear_weights(binding, "mtp.layers.0.mlp.gate_proj",
                                               kHidden, kIntermediate),
                                linear_metadata(kHidden, kIntermediate), post_norm,
                                gate_projection, 1, stream,
                                Exl3CudaLinearAdmission::native_mtp_one_step);
        projections[6]->forward(linear_weights(binding, "mtp.layers.0.mlp.up_proj",
                                               kHidden, kIntermediate),
                                linear_metadata(kHidden, kIntermediate), post_norm,
                                up_projection, 1, stream,
                                Exl3CudaLinearAdmission::native_mtp_one_step);
        silu_mul_kernel<<<(kIntermediate + 255) / 256, 256, 0, stream>>>(
            gate_projection, up_projection, activated);
        check(cudaGetLastError(), "launch native MTP SwiGLU");
        projections[7]->forward(linear_weights(binding, "mtp.layers.0.mlp.down_proj",
                                               kIntermediate, kHidden),
                                linear_metadata(kIntermediate, kHidden), activated,
                                output_residual, 1, stream,
                                Exl3CudaLinearAdmission::native_mtp_one_step);
        residual_kernel<<<(kHidden + 255) / 256, 256, 0, stream>>>(
            output_residual, attention_input,
            reinterpret_cast<std::uint16_t*>(output.hidden.data), kHidden);
        check(cudaGetLastError(), "launch native MTP final residual");
        rmsnorm_bf16_weight_kernel<<<1, 512, 0, stream>>>(
            reinterpret_cast<const std::uint16_t*>(output.hidden.data),
            norm("mtp.norm.weight"), reinterpret_cast<std::uint16_t*>(output.hidden.data), kHidden);
        check(cudaGetLastError(), "launch native MTP final norm");
        projections[8]->forward(binding.shared_target.lm_head_weights,
                                 binding.shared_target.lm_head_metadata,
                                 reinterpret_cast<const std::uint16_t*>(output.hidden.data),
                                 reinterpret_cast<std::uint16_t*>(output.logits.data), 1, stream);
        argmax_kernel<<<1, 1, 0, stream>>>(
            reinterpret_cast<const std::uint16_t*>(output.logits.data),
            reinterpret_cast<std::int32_t*>(output.predicted_token.data));
        check(cudaGetLastError(), "launch native MTP target-head argmax");
        return true;
    }

    bool set_mtp_kv_valid_rows(std::uint32_t rows) noexcept {
        if (!admitted || !binding.mtp_kv.valid() || rows > binding.mtp_kv.capacity) return false;
        binding.mtp_kv.valid_rows = rows;
        return true;
    }
};

Exl3NativeMtpOneStepExecutor::Exl3NativeMtpOneStepExecutor(
    Exl3MtpExecutionBinding binding)
    : impl_(std::make_unique<Impl>(std::move(binding))) {}

Exl3NativeMtpOneStepExecutor::~Exl3NativeMtpOneStepExecutor() = default;

bool Exl3NativeMtpOneStepExecutor::valid() const noexcept {
    return impl_ && impl_->admitted;
}

const Exl3MtpExecutionBinding& Exl3NativeMtpOneStepExecutor::binding() const noexcept {
    return impl_->binding;
}

bool Exl3NativeMtpOneStepExecutor::set_mtp_kv_valid_rows(std::uint32_t rows) noexcept {
    return impl_ && impl_->set_mtp_kv_valid_rows(rows);
}

bool Exl3NativeMtpOneStepExecutor::enqueue(const Exl3MtpStepInput& input,
                                           const Exl3MtpStepOutput& output,
                                           std::uintptr_t retained_stream) noexcept {
    if (!impl_) return false;
    try {
        return impl_->enqueue(input, output, retained_stream);
    } catch (...) {
        // enqueue() may have queued earlier stages before a later validation,
        // launch, or linear submission throws.  The caller treats false as an
        // immediate ordinary-target fallback, so drain this stream before it
        // can release the input/output owners; otherwise a partial MTP graph
        // could still dereference storage that the fallback reuses.
        (void)cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(retained_stream));
        return false;
    }
}

bool Exl3NativeMtpOneStepExecutor::enqueue_callback(
    const Exl3MtpExecutionBinding& binding, const Exl3MtpStepInput& input,
    const Exl3MtpStepOutput& output, std::uintptr_t retained_stream,
    void* user) noexcept {
    auto* executor = static_cast<Exl3NativeMtpOneStepExecutor*>(user);
    if (!executor || executor->binding().model_identity != binding.model_identity)
        return false;
    return executor->enqueue(input, output, retained_stream);
}

namespace {

struct PrefixDeviceBuffer {
    void* data = nullptr;
    std::size_t bytes = 0;

    PrefixDeviceBuffer(std::size_t extent, const char* label) : bytes(extent) {
        check(cudaMalloc(&data, bytes), label);
    }
    ~PrefixDeviceBuffer() {
        if (data != nullptr) (void)cudaFree(data);
    }
    PrefixDeviceBuffer(const PrefixDeviceBuffer&) = delete;
    PrefixDeviceBuffer& operator=(const PrefixDeviceBuffer&) = delete;
};

constexpr std::size_t kPrefixKvRowBytes = 1'024U * sizeof(std::uint16_t);
constexpr std::size_t kPrefixHiddenBytes =
    static_cast<std::size_t>(kExl3MtpHiddenWidth) * sizeof(std::uint16_t);
constexpr std::size_t kPrefixLogitsBytes =
    static_cast<std::size_t>(kExl3MtpVocabSize) * sizeof(std::uint16_t);

} // namespace

struct Exl3MtpPrefixState::Impl {
    Exl3MtpBinding manifest;
    std::vector<Exl3MtpDeviceTensor> tensors;
    std::vector<std::unique_ptr<PrefixDeviceBuffer>> weights;
    std::unique_ptr<PrefixDeviceBuffer> key;
    std::unique_ptr<PrefixDeviceBuffer> value;
    std::array<std::unique_ptr<PrefixDeviceBuffer>, 2> hidden;
    std::unique_ptr<PrefixDeviceBuffer> logits;
    std::unique_ptr<PrefixDeviceBuffer> predicted_token;
    Exl3MtpExecutionBinding binding;
    std::shared_ptr<const void> owner_token;
    std::unique_ptr<Exl3NativeMtpOneStepExecutor> executor;
    std::vector<std::shared_ptr<const void>> handoff_owners;
    cudaEvent_t ready_event = nullptr;
    std::uintptr_t last_stream = 0;
    bool ready_recorded = false;
    std::uint64_t hidden_generation = 0;
    int last_hidden_index = -1;
    bool admitted = false;
    bool poisoned = false;

    ~Impl() {
        if (ready_event != nullptr) {
            if (ready_recorded) {
                (void)cudaEventSynchronize(ready_event);
            } else if (last_stream != 0) {
                (void)cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(last_stream));
            }
            (void)cudaEventDestroy(ready_event);
            ready_event = nullptr;
        }
    }

    [[nodiscard]] Exl3MtpBuffer hidden_view(int index) const noexcept {
        if (index < 0 || index >= static_cast<int>(hidden.size()) || !hidden[index]) return {};
        return {reinterpret_cast<std::uintptr_t>(hidden[index]->data),
                hidden[index]->bytes, kExl3MtpHiddenWidth, 1};
    }

    [[nodiscard]] Exl3MtpBuffer logits_view() const noexcept {
        if (!logits) return {};
        return {reinterpret_cast<std::uintptr_t>(logits->data), logits->bytes,
                kExl3MtpVocabSize, 1};
    }

    [[nodiscard]] Exl3MtpBuffer token_view() const noexcept {
        if (!predicted_token) return {};
        return {reinterpret_cast<std::uintptr_t>(predicted_token->data),
                predicted_token->bytes, 1, 1};
    }

    bool record_ready(std::uintptr_t retained_stream) noexcept {
        if (ready_event == nullptr || retained_stream == 0) return false;
        const bool recorded = cudaEventRecord(
                                  ready_event, reinterpret_cast<cudaStream_t>(retained_stream)) ==
                              cudaSuccess;
        ready_recorded = recorded;
        return recorded;
    }
};

Exl3MtpPrefixState::Exl3MtpPrefixState(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

Exl3MtpPrefixState::~Exl3MtpPrefixState() = default;

std::shared_ptr<Exl3MtpPrefixState> Exl3MtpPrefixState::materialize(
    const IndexedSafetensors& collection, Exl3MtpSharedTargetBinding shared_target,
    std::uint64_t model_identity, std::uint32_t kv_capacity) {
    if (model_identity == 0 || kv_capacity == 0 || kv_capacity > 32'768U ||
        !shared_target.valid() || shared_target.lm_head_weights.trellis == nullptr ||
        shared_target.lm_head_weights.suh == nullptr ||
        shared_target.lm_head_weights.svh == nullptr ||
        shared_target.lm_head_weights.mul1 == nullptr ||
        shared_target.lm_head_metadata.K != 6 ||
        shared_target.lm_head_metadata.in_features != static_cast<int>(kExl3MtpHiddenWidth) ||
        shared_target.lm_head_metadata.out_features != static_cast<int>(kExl3MtpVocabSize)) {
        throw std::invalid_argument("native MTP prefix target binding is not admitted");
    }
    auto state = std::shared_ptr<Exl3MtpPrefixState>(
        new Exl3MtpPrefixState(std::make_unique<Impl>()));
    auto& impl = *state->impl_;
    impl.owner_token = std::shared_ptr<const void>(state,
                                                   static_cast<const void*>(state.get()));
    impl.manifest = bind_qwen3_8_27b_native_mtp(collection);
    impl.tensors.reserve(impl.manifest.tensors.size());
    impl.weights.reserve(impl.manifest.tensors.size());

    const auto specs = qwen3_8_27b_native_mtp_manifest();
    for (const auto& descriptor : impl.manifest.tensors) {
        const auto spec = std::find_if(specs.begin(), specs.end(), [&](const auto& candidate) {
            return candidate.name == descriptor.name;
        });
        if (spec == specs.end()) {
            throw SafetensorsError("native MTP materialization encountered an unknown tensor");
        }
        const auto payload = read_tensor(descriptor.shard_path, descriptor.name);
        if (payload.info.dtype != descriptor.info.dtype ||
            payload.info.shape != descriptor.info.shape ||
            payload.bytes().size() != spec->bytes) {
            throw SafetensorsError("native MTP tensor payload changed after manifest binding: " +
                                   std::string(descriptor.name));
        }
        auto allocation = std::make_unique<PrefixDeviceBuffer>(
            static_cast<std::size_t>(spec->bytes), "allocate native MTP manifest tensor");
        check(cudaMemcpy(allocation->data, payload.bytes().data(), payload.bytes().size(),
                         cudaMemcpyHostToDevice),
              "upload native MTP manifest tensor");
        impl.tensors.push_back({descriptor.name,
                                reinterpret_cast<std::uintptr_t>(allocation->data),
                                allocation->bytes});
        impl.weights.push_back(std::move(allocation));
    }

    impl.key = std::make_unique<PrefixDeviceBuffer>(
        static_cast<std::size_t>(kv_capacity) * kPrefixKvRowBytes,
        "allocate native MTP prefix key cache");
    impl.value = std::make_unique<PrefixDeviceBuffer>(
        static_cast<std::size_t>(kv_capacity) * kPrefixKvRowBytes,
        "allocate native MTP prefix value cache");
    impl.hidden[0] = std::make_unique<PrefixDeviceBuffer>(
        kPrefixHiddenBytes, "allocate native MTP prefix hidden 0");
    impl.hidden[1] = std::make_unique<PrefixDeviceBuffer>(
        kPrefixHiddenBytes, "allocate native MTP prefix hidden 1");
    impl.logits = std::make_unique<PrefixDeviceBuffer>(
        kPrefixLogitsBytes, "allocate native MTP prefix logits");
    impl.predicted_token = std::make_unique<PrefixDeviceBuffer>(
        sizeof(std::int32_t), "allocate native MTP prefix token");
    check(cudaEventCreateWithFlags(&impl.ready_event, cudaEventDisableTiming),
          "create native MTP prefix ready event");

    impl.binding.manifest = &impl.manifest;
    impl.binding.tensors = std::span<const Exl3MtpDeviceTensor>(impl.tensors.data(),
                                                                 impl.tensors.size());
    impl.binding.shared_target = std::move(shared_target);
    impl.binding.mtp_kv = {
        {reinterpret_cast<std::uintptr_t>(impl.key->data), impl.key->bytes, kv_capacity,
         1'024U},
        {reinterpret_cast<std::uintptr_t>(impl.value->data), impl.value->bytes, kv_capacity,
         1'024U},
        kv_capacity,
        0,
        impl.owner_token};
    impl.binding.weights_owner = impl.owner_token;
    impl.binding.model_identity = model_identity;
    impl.handoff_owners.reserve(kv_capacity);
    if (!impl.binding.valid()) {
        throw std::invalid_argument("native MTP prefix binding failed strict validation");
    }
    impl.executor = std::make_unique<Exl3NativeMtpOneStepExecutor>(impl.binding);
    if (!impl.executor->valid()) {
        throw std::invalid_argument("native MTP prefix executor was not admitted");
    }
    impl.admitted = true;
    return state;
}

bool Exl3MtpPrefixState::valid() const noexcept {
    return impl_ && impl_->admitted && !impl_->poisoned && impl_->binding.valid();
}

const Exl3MtpExecutionBinding& Exl3MtpPrefixState::binding() const noexcept {
    static const Exl3MtpExecutionBinding empty{};
    return impl_ ? impl_->binding : empty;
}

std::shared_ptr<const void> Exl3MtpPrefixState::owner() const noexcept {
    return impl_ ? impl_->owner_token : std::shared_ptr<const void>{};
}

Exl3MtpBuffer Exl3MtpPrefixState::last_hidden() const noexcept {
    return impl_ ? impl_->hidden_view(impl_->last_hidden_index) : Exl3MtpBuffer{};
}

Exl3MtpBuffer Exl3MtpPrefixState::last_logits() const noexcept {
    return impl_ ? impl_->logits_view() : Exl3MtpBuffer{};
}

Exl3MtpBuffer Exl3MtpPrefixState::last_predicted_token() const noexcept {
    return impl_ ? impl_->token_view() : Exl3MtpBuffer{};
}

Exl3MtpPrefixReceipt Exl3MtpPrefixState::prefill_target_prefix(
    std::span<const Exl3MtpTargetHiddenHandoff> hidden_rows,
    std::span<const std::int64_t> token_ids, std::uint64_t request_identity,
    std::uint64_t target_identity, std::uintptr_t retained_stream) noexcept {
    Exl3MtpPrefixReceipt receipt{};
    if (!impl_ || !valid() || hidden_rows.empty() ||
        token_ids.size() != hidden_rows.size() + 1U || request_identity == 0 ||
        target_identity == 0 || retained_stream == 0 ||
        (impl_->last_stream != 0 && impl_->last_stream != retained_stream)) {
        return receipt;
    }
    const std::uint32_t start = impl_->binding.mtp_kv.valid_rows;
    if (hidden_rows.size() > impl_->binding.mtp_kv.capacity - start) return receipt;
    const auto first_position = hidden_rows.front().position;
    if (first_position < 0 || first_position != static_cast<std::int64_t>(start)) return receipt;

    const auto model_owner = impl_->binding.shared_target.model_identity
                                 ? impl_->binding.shared_target.model_identity.get()
                                 : impl_->binding.shared_target.owner.get();
    const auto generation = hidden_rows.front().hidden_generation;
    if (generation == 0) return receipt;
    for (std::size_t i = 0; i < hidden_rows.size(); ++i) {
        const auto& row = hidden_rows[i];
        if (!row.valid() || row.model_identity.get() != model_owner ||
            row.hidden_generation != generation ||
            row.position != first_position + static_cast<std::int64_t>(i)) {
            return receipt;
        }
        const auto token = token_ids[i + 1U];
        if (token < 0 || token >= static_cast<std::int64_t>(kExl3MtpTokenDomain)) return receipt;
        impl_->handoff_owners.push_back(row.owner);
    }

    impl_->last_stream = retained_stream;
    impl_->hidden_generation = generation;
    for (std::size_t i = 0; i < hidden_rows.size(); ++i) {
        const auto& row = hidden_rows[i];
        const int output_index = static_cast<int>((impl_->last_hidden_index + 1) & 1);
        Exl3MtpStepInput input{};
        input.request_identity = request_identity;
        input.target_identity = target_identity;
        input.model_identity = impl_->binding.model_identity;
        input.hidden_generation = generation;
        input.position = row.position;
        input.token_position = row.position + 1;
        input.token = static_cast<std::int32_t>(token_ids[i + 1U]);
        // Prefill is deliberately teacher-forced from the target's exact
        // final-normalized row for every position.  Reusing the preceding
        // MTP hidden here would silently turn prompt prefill into the
        // autoregressive path and does not match the native MTP contract.
        input.seed_kind = Exl3MtpSeedKind::TargetHidden;
        input.hidden_owner = row.owner;
        input.hidden = row.hidden;
        const Exl3MtpStepOutput output{impl_->hidden_view(output_index),
                                       impl_->logits_view(), impl_->token_view()};
        if (!input.valid() || !impl_->executor->enqueue(input, output, retained_stream) ||
            !impl_->executor->set_mtp_kv_valid_rows(start + static_cast<std::uint32_t>(i) + 1U)) {
            impl_->poisoned = true;
            (void)impl_->record_ready(retained_stream);
            return receipt;
        }
        impl_->binding.mtp_kv.valid_rows =
            start + static_cast<std::uint32_t>(i) + 1U;
        impl_->last_hidden_index = output_index;
    }
    if (!impl_->record_ready(retained_stream)) {
        impl_->poisoned = true;
        return receipt;
    }
    receipt.owner = impl_->owner_token;
    receipt.request_identity = request_identity;
    receipt.target_identity = target_identity;
    receipt.model_identity = impl_->binding.model_identity;
    receipt.hidden_generation = generation;
    receipt.valid_rows = impl_->binding.mtp_kv.valid_rows;
    receipt.first_position = first_position;
    receipt.last_position = hidden_rows.back().position;
    receipt.producer_event = reinterpret_cast<std::uintptr_t>(impl_->ready_event);
    return receipt;
}

Exl3MtpPrefixReceipt Exl3MtpPrefixState::ar_step(
    std::int64_t token, std::uint64_t request_identity,
    std::uint64_t target_identity, std::uintptr_t retained_stream) noexcept {
    Exl3MtpPrefixReceipt receipt{};
    if (!impl_ || !valid() || impl_->last_hidden_index < 0 || token < 0 ||
        token >= static_cast<std::int64_t>(kExl3MtpTokenDomain) || request_identity == 0 ||
        target_identity == 0 || retained_stream == 0 ||
        (impl_->last_stream != 0 && impl_->last_stream != retained_stream)) {
        return receipt;
    }
    const std::uint32_t position = impl_->binding.mtp_kv.valid_rows;
    if (position >= impl_->binding.mtp_kv.capacity || impl_->hidden_generation == 0) return receipt;
    const int output_index = static_cast<int>((impl_->last_hidden_index + 1) & 1);
    Exl3MtpStepInput input{};
    input.request_identity = request_identity;
    input.target_identity = target_identity;
    input.model_identity = impl_->binding.model_identity;
    input.hidden_generation = impl_->hidden_generation;
    input.position = static_cast<std::int64_t>(position);
    input.token_position = input.position + 1;
    input.token = static_cast<std::int32_t>(token);
    input.seed_kind = Exl3MtpSeedKind::PreviousMtpHidden;
    input.hidden_owner = impl_->owner_token;
    input.hidden = impl_->hidden_view(impl_->last_hidden_index);
    const Exl3MtpStepOutput output{impl_->hidden_view(output_index),
                                   impl_->logits_view(), impl_->token_view()};
    if (!input.valid() || !impl_->executor->enqueue(input, output, retained_stream) ||
        !impl_->executor->set_mtp_kv_valid_rows(position + 1U)) {
        impl_->poisoned = true;
        (void)impl_->record_ready(retained_stream);
        return receipt;
    }
    impl_->binding.mtp_kv.valid_rows = position + 1U;
    impl_->last_hidden_index = output_index;
    impl_->last_stream = retained_stream;
    if (!impl_->record_ready(retained_stream)) {
        impl_->poisoned = true;
        return receipt;
    }
    receipt.owner = impl_->owner_token;
    receipt.request_identity = request_identity;
    receipt.target_identity = target_identity;
    receipt.model_identity = impl_->binding.model_identity;
    receipt.hidden_generation = impl_->hidden_generation;
    receipt.valid_rows = impl_->binding.mtp_kv.valid_rows;
    receipt.first_position = input.position;
    receipt.last_position = input.position;
    receipt.producer_event = reinterpret_cast<std::uintptr_t>(impl_->ready_event);
    return receipt;
}

bool Exl3MtpPrefixState::reset() noexcept {
    if (!impl_) return false;
    cudaError_t synchronization = cudaSuccess;
    if (impl_->ready_event != nullptr && impl_->ready_recorded) {
        synchronization = cudaEventSynchronize(impl_->ready_event);
    } else if (impl_->last_stream != 0) {
        synchronization = cudaStreamSynchronize(
            reinterpret_cast<cudaStream_t>(impl_->last_stream));
    }
    if (synchronization != cudaSuccess) {
        impl_->poisoned = true;
        return false;
    }
    impl_->handoff_owners.clear();
    impl_->binding.mtp_kv.valid_rows = 0;
    if (impl_->executor && !impl_->executor->set_mtp_kv_valid_rows(0)) return false;
    impl_->last_stream = 0;
    impl_->ready_recorded = false;
    impl_->hidden_generation = 0;
    impl_->last_hidden_index = -1;
    impl_->poisoned = false;
    return true;
}

} // namespace ninfer::exl3
