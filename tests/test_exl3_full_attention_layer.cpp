#include "exl3/full_attention_layer.h"
#include "exl3/safetensors.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;
using ninfer::exl3::Exl3CudaLinearMetadata;
using ninfer::exl3::Exl3CudaLinearWorkspace;
using ninfer::exl3::Exl3CudaLinearWeights;
using ninfer::exl3::Exl3FullAttentionLayer;
using ninfer::exl3::Exl3FullAttentionLayerTrace;
using ninfer::exl3::Exl3FullAttentionLayerTimings;
using ninfer::exl3::Exl3FullAttentionLayerWeights;
using ninfer::exl3::Exl3LayerBufferRetirement;
using ninfer::exl3::IndexedSafetensors;
using ninfer::exl3::SafetensorsHeader;
using ninfer::exl3::TensorPayload;

constexpr int kHidden = 5120;
constexpr int kQProjection = 12288;
constexpr int kKVProjection = 1024;
constexpr int kIntermediate = 17408;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string env_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
}

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

struct DeviceBuffer {
    void* ptr = nullptr;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

std::unique_ptr<DeviceBuffer> upload(std::span<const std::byte> bytes, const char* label) {
    auto buffer = std::make_unique<DeviceBuffer>();
    cuda_check(cudaMalloc(&buffer->ptr, bytes.size_bytes()), label);
    try {
        cuda_check(cudaMemcpy(buffer->ptr, bytes.data(), bytes.size_bytes(), cudaMemcpyHostToDevice),
                   "upload EXL3 E3A tensor");
    } catch (...) {
        cudaFree(buffer->ptr);
        buffer->ptr = nullptr;
        throw;
    }
    return buffer;
}

TensorPayload load_indexed_tensor(const IndexedSafetensors& collection,
                                  const std::string& name) {
    for (const auto& shard : collection.shards) {
        if (shard.header.find(name) != nullptr) return ninfer::exl3::read_tensor(shard.path, shard.header, name);
    }
    throw std::runtime_error("indexed tensor is absent: " + name);
}

float bf16_to_float(std::uint16_t bits) {
    std::uint32_t value = static_cast<std::uint32_t>(bits) << 16u;
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

std::uint16_t float_to_half(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    const std::uint32_t exponent = (bits >> 23u) & 0xffu;
    const std::uint32_t fraction = bits & 0x007fffffu;
    if (exponent == 0xffu) return static_cast<std::uint16_t>(sign | (fraction ? 0x7e00u : 0x7c00u));
    const int unbiased = static_cast<int>(exponent) - 127;
    if (unbiased > 15) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (unbiased >= -14) {
        std::uint32_t half_exp = static_cast<std::uint32_t>(unbiased + 15);
        std::uint32_t half_frac = fraction >> 13u;
        const std::uint32_t remainder = fraction & 0x1fffu;
        if (remainder > 0x1000u || (remainder == 0x1000u && (half_frac & 1u))) {
            ++half_frac;
            if (half_frac == 0x400u) { half_frac = 0; ++half_exp; }
        }
        if (half_exp >= 0x1fu) return static_cast<std::uint16_t>(sign | 0x7c00u);
        return static_cast<std::uint16_t>(sign | (half_exp << 10u) | half_frac);
    }
    if (unbiased < -25) return static_cast<std::uint16_t>(sign);
    const std::uint32_t mantissa = fraction | 0x00800000u;
    const int shift = -unbiased - 14;
    std::uint32_t half_frac = mantissa >> (shift + 13);
    const std::uint32_t remainder_mask = (1u << (shift + 13)) - 1u;
    const std::uint32_t remainder = mantissa & remainder_mask;
    const std::uint32_t halfway = 1u << (shift + 12);
    if (remainder > halfway || (remainder == halfway && (half_frac & 1u))) ++half_frac;
    return static_cast<std::uint16_t>(sign | half_frac);
}

std::vector<std::uint16_t> norm_f16(const TensorPayload& tensor) {
    require(tensor.info.dtype == "BF16", tensor.info.name + " must be BF16");
    const auto source = tensor.typed<std::uint16_t>("BF16");
    std::vector<std::uint16_t> result(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) result[i] = float_to_half(bf16_to_float(source[i]));
    return result;
}

struct LinearStorage {
    Exl3CudaLinearWeights view{};
    Exl3CudaLinearMetadata metadata{};
    std::vector<TensorPayload> host;
};

LinearStorage make_linear(const IndexedSafetensors& collection,
                          const std::string& prefix,
                          int in_features,
                          int out_features,
                          std::vector<std::unique_ptr<DeviceBuffer>>& device) {
    LinearStorage result;
    auto load = [&](const char* suffix) {
        result.host.push_back(load_indexed_tensor(collection, prefix + "." + suffix));
        return result.host.back();
    };
    const auto trellis = load("trellis");
    const auto suh = load("suh");
    const auto svh = load("svh");
    const auto mul1 = load("mul1");
    require(trellis.info.dtype == "I16" && trellis.info.shape.size() == 3, prefix + " trellis metadata mismatch");
    require(suh.info.dtype == "F16" && svh.info.dtype == "F16" && mul1.info.dtype == "I32",
            prefix + " auxiliary metadata mismatch");
    device.emplace_back(upload(trellis.bytes(), "upload EXL3 trellis"));
    result.view.trellis = static_cast<const std::uint16_t*>(device.back()->ptr);
    device.emplace_back(upload(suh.bytes(), "upload EXL3 suh"));
    result.view.suh = static_cast<const std::uint16_t*>(device.back()->ptr);
    device.emplace_back(upload(svh.bytes(), "upload EXL3 svh"));
    result.view.svh = static_cast<const std::uint16_t*>(device.back()->ptr);
    device.emplace_back(upload(mul1.bytes(), "upload EXL3 mul1"));
    result.view.mul1 = static_cast<const std::int32_t*>(device.back()->ptr);
    result.metadata = {in_features, out_features,
                       static_cast<int>(trellis.info.shape[2] / 16), false, true, false};
    require(trellis.info.shape[0] == static_cast<std::uint64_t>(in_features / 16) &&
            trellis.info.shape[1] == static_cast<std::uint64_t>(out_features / 16),
            prefix + " trellis dimensions do not match the logical projection");
    return result;
}

struct Metrics {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rms = 0.0;
    double relative_l2 = 0.0;
    double p99 = 0.0;
    double p999 = 0.0;
    std::size_t worst_index = 0;
};

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (bits & 0x8000u) << 16u;
    const std::uint32_t exponent = (bits >> 10u) & 0x1fu;
    const std::uint32_t fraction = bits & 0x03ffu;
    std::uint32_t value = sign;
    if (exponent == 0) {
        if (fraction) {
            std::uint32_t normalized = fraction;
            int shift = 0;
            while ((normalized & 0x0400u) == 0) { normalized <<= 1u; ++shift; }
            value |= static_cast<std::uint32_t>(127 - 14 - shift) << 23u;
            value |= (normalized & 0x03ffu) << 13u;
        }
    } else if (exponent == 0x1fu) value |= 0x7f800000u | (fraction << 13u);
    else value |= (exponent + 112u) << 23u | (fraction << 13u);
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

Metrics compare(std::span<const std::uint16_t> actual,
                std::span<const std::uint16_t> expected) {
    require(actual.size() == expected.size(), "E3A stage size mismatch");
    std::vector<double> errors;
    errors.reserve(actual.size());
    double sum = 0.0, squares = 0.0, expected_squares = 0.0;
    Metrics result;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(static_cast<double>(half_to_float(actual[i])) - half_to_float(expected[i]));
        errors.push_back(error);
        sum += error; squares += error * error;
        const double expected_value = half_to_float(expected[i]);
        expected_squares += expected_value * expected_value;
        if (error > result.max_abs) { result.max_abs = error; result.worst_index = i; }
    }
    std::sort(errors.begin(), errors.end());
    const auto percentile = [&](double p) { return errors[std::min(errors.size() - 1, static_cast<std::size_t>(p * (errors.size() - 1)))]; };
    result.mean_abs = sum / actual.size();
    result.rms = std::sqrt(squares / actual.size());
    result.relative_l2 = expected_squares == 0.0 ? std::sqrt(squares) : std::sqrt(squares / expected_squares);
    result.p99 = percentile(0.99); result.p999 = percentile(0.999);
    return result;
}

std::vector<std::uint16_t> tensor_as_f16(const TensorPayload& tensor) {
    if (tensor.info.dtype == "F16") {
        const auto values = tensor.typed<std::uint16_t>("F16");
        return std::vector<std::uint16_t>(values.begin(), values.end());
    }
    require(tensor.info.dtype == "F32", tensor.info.name + " must be F16 or F32");
    const auto values = tensor.typed<float>("F32");
    std::vector<std::uint16_t> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) result[i] = float_to_half(values[i]);
    return result;
}

struct Stage { const char* label; std::string fixture; const std::uint16_t* actual; };

std::vector<Stage> stages(const Exl3FullAttentionLayerTrace& trace, const std::string& tag) {
    return {
        {"layer input", (tag + "_layer_output_input").c_str(), trace.layer_input},
        {"input norm", (tag + "_input_norm").c_str(), trace.input_norm},
        {"Q projection", (tag + "_q_projection").c_str(), trace.q_projection},
        {"K projection", (tag + "_k_projection").c_str(), trace.k_projection},
        {"V projection", (tag + "_v_projection").c_str(), trace.v_projection},
        {"Q norm", (tag + "_q_normed").c_str(), trace.q_normed},
        {"K norm", (tag + "_k_normed").c_str(), trace.k_normed},
        {"RoPE Q", (tag + "_q_rope").c_str(), trace.q_rope},
        {"RoPE K", (tag + "_k_rope").c_str(), trace.k_rope},
        {"attention output", (tag + "_attention_output").c_str(), trace.attention_output},
        {"output projection", (tag + "_output_projection").c_str(), trace.output_projection},
        {"post-attention residual", (tag + "_post_attention_residual").c_str(), trace.post_attention_residual},
        {"MLP input norm", (tag + "_mlp_input").c_str(), trace.mlp_input},
        {"gate projection", (tag + "_gate_projection").c_str(), trace.gate_projection},
        {"up projection", (tag + "_up_projection").c_str(), trace.up_projection},
        {"activated MLP", (tag + "_activated_mlp").c_str(), trace.activated_mlp},
        {"down projection", (tag + "_down_projection").c_str(), trace.down_projection},
        {"layer output", (tag + "_layer_output").c_str(), trace.layer_output},
    };
}

void compare_trace(const Exl3FullAttentionLayerTrace& trace,
                   const SafetensorsHeader& fixture,
                   const std::string& tag,
                   std::vector<Metrics>& metrics_out) {
    for (const auto& stage : stages(trace, tag)) {
        const auto* info = fixture.find(stage.fixture);
        require(info != nullptr, std::string("fixture stage missing: ") + stage.fixture);
        const auto expected = ninfer::exl3::read_tensor(
            std::filesystem::path(env_or_empty("NINFER_EXL3_ORACLE_PATH")), fixture, stage.fixture);
        const auto expected_values = tensor_as_f16(expected);
        std::vector<std::uint16_t> actual(expected_values.size());
        cuda_check(cudaMemcpy(actual.data(), stage.actual, actual.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                   "download E3A stage");
        const auto metric = compare(actual, expected_values);
        metrics_out.push_back(metric);
        std::cout << "STAGE " << tag << " " << stage.label
                  << " max_abs=" << std::setprecision(10) << metric.max_abs
                  << " mean_abs=" << metric.mean_abs << " rms=" << metric.rms
                  << " relative_l2=" << metric.relative_l2
                  << " p99=" << metric.p99 << " p999=" << metric.p999
                  << " worst_index=" << metric.worst_index
                  << " actual=" << half_to_float(actual[metric.worst_index])
                  << " expected=" << half_to_float(expected_values[metric.worst_index]) << '\n';
    }
}

Exl3FullAttentionLayerTimings average_timings(const std::vector<Exl3FullAttentionLayerTimings>& values) {
    Exl3FullAttentionLayerTimings result;
    for (const auto& value : values) {
        for (std::size_t i = 0; i < result.microseconds.size(); ++i) result.microseconds[i] += value.microseconds[i];
        result.total_microseconds += value.total_microseconds;
    }
    for (double& value : result.microseconds) value /= static_cast<double>(values.size());
    result.total_microseconds /= static_cast<double>(values.size());
    return result;
}

} // namespace

int main() {
    try {
        // Independent storage ledger: seven transforms, five output splits,
        // seventeen FP16 planes; shared scratch retains only V/rotated Q/K.
        for(int rows:{1,14,16,128,1024})for(bool accum:{false,true})
            for(bool transform:{false,true})for(bool scratch:{false,true})
                for(bool alias:{false,true}) {
                    const auto linear=(transform?0ull:49152ull*2)+(accum?0ull:59392ull*5*4);
                    const auto planes=scratch?8192ull:117760ull-(alias?5120ull:0ull);
                    const auto expected=static_cast<std::size_t>(rows)*(linear+planes*2);
                    require(Exl3FullAttentionLayer::workspace_bytes_required(rows,accum,transform,scratch,alias)==expected,
                        "independent full-attention requirement ledger");
                }
        for(int rows:{-1,0,1025}) {
            bool refused=false;
            try{(void)Exl3FullAttentionLayer::workspace_bytes_required(rows,false,false,false,false);}
            catch(const std::invalid_argument&){refused=true;}
            require(refused,"full-attention requirement accepted unsupported rows");
        }
        const auto target_path = env_or_empty("NINFER_EXL3_TARGET_PATH");
        const auto fixture_path = env_or_empty("NINFER_EXL3_ORACLE_PATH");
        if (target_path.empty() || fixture_path.empty()) {
            std::cerr << "E3A skipped: set NINFER_EXL3_TARGET_PATH and NINFER_EXL3_ORACLE_PATH\n";
            return 77;
        }
        const auto collection = ninfer::exl3::inspect_indexed_directory(target_path);
        const auto fixture = ninfer::exl3::inspect_file(fixture_path);
        std::vector<std::unique_ptr<DeviceBuffer>> device;
        const std::string base = "model.language_model.layers.19.";
        auto q = make_linear(collection, base + "self_attn.q_proj", kHidden, kQProjection, device);
        auto k = make_linear(collection, base + "self_attn.k_proj", kHidden, kKVProjection, device);
        auto v = make_linear(collection, base + "self_attn.v_proj", kHidden, kKVProjection, device);
        auto o = make_linear(collection, base + "self_attn.o_proj", kQProjection / 2, kHidden, device);
        auto gate = make_linear(collection, base + "mlp.gate_proj", kHidden, kIntermediate, device);
        auto up = make_linear(collection, base + "mlp.up_proj", kHidden, kIntermediate, device);
        auto down = make_linear(collection, base + "mlp.down_proj", kIntermediate, kHidden, device);

        auto load_norm = [&](const std::string& name) {
            auto host = norm_f16(load_indexed_tensor(collection, name));
            const auto bytes = std::span<const std::byte>(reinterpret_cast<const std::byte*>(host.data()), host.size() * sizeof(std::uint16_t));
            auto buffer = upload(bytes, "upload EXL3 E3A norm");
            const auto* ptr = static_cast<const std::uint16_t*>(buffer->ptr);
            device.emplace_back(std::move(buffer));
            return ptr;
        };
        Exl3FullAttentionLayerWeights weights;
        weights.q = q.view; weights.k = k.view; weights.v = v.view; weights.o = o.view;
        weights.gate = gate.view; weights.up = up.view; weights.down = down.view;
        weights.q_metadata = q.metadata; weights.k_metadata = k.metadata; weights.v_metadata = v.metadata;
        weights.o_metadata = o.metadata; weights.gate_metadata = gate.metadata;
        weights.up_metadata = up.metadata; weights.down_metadata = down.metadata;
        weights.input_norm = load_norm(base + "input_layernorm.weight");
        weights.q_norm = load_norm(base + "self_attn.q_norm.weight");
        weights.k_norm = load_norm(base + "self_attn.k_norm.weight");
        weights.post_attention_norm = load_norm(base + "post_attention_layernorm.weight");

        if(const auto* mode=std::getenv("NINFER_TEST_ATTENTION_COALESCED_PARITY");mode && std::string_view(mode)=="1") {
            const auto tensor=ninfer::exl3::read_tensor(fixture_path,fixture,"prefill14_layer_output_input");
            const auto original=tensor_as_f16(tensor);
            std::vector<std::uint16_t> input(16u*kHidden);
            for(std::size_t i=0;i<input.size();++i)input[i]=original[i%original.size()];
            DeviceBuffer in,control_out,candidate_out;
            cuda_check(cudaMalloc(&in.ptr,input.size()*2),"coalesced input allocation");
            cuda_check(cudaMalloc(&control_out.ptr,input.size()*2),"coalesced control allocation");
            cuda_check(cudaMalloc(&candidate_out.ptr,input.size()*2),"coalesced output allocation");
            cuda_check(cudaMemcpy(in.ptr,input.data(),input.size()*2,cudaMemcpyHostToDevice),"coalesced input upload");
            for(bool shared:{false,true}) {
                DeviceBuffer control_scratch,candidate_scratch;
                ninfer::exl3::Exl3CudaLayerScratchView a{},b{};
                if(shared) {
                    a.bytes=Exl3FullAttentionLayer::shared_scratch_bytes(16);
                    b.bytes=Exl3FullAttentionLayer::shared_scratch_bytes(16,true);
                    cuda_check(cudaMalloc(&control_scratch.ptr,a.bytes),"control shared scratch");
                    cuda_check(cudaMalloc(&candidate_scratch.ptr,b.bytes),"coalesced shared scratch");
                    a.data=control_scratch.ptr;b.data=candidate_scratch.ptr;
                }
                Exl3FullAttentionLayer control(weights,16,{}, {},a);
                Exl3FullAttentionLayer candidate(weights,16,{}, {},b,nullptr,true);
                require(control.fixed_owner_metadata_bytes()==Exl3FullAttentionLayer::fixed_owner_metadata_required() &&
                    candidate.fixed_owner_metadata_bytes()==control.fixed_owner_metadata_bytes(),
                    "attention borrowed/coalesced storage changed fixed owner metadata");
                for(int rows:{1,8,14,16}) {
                    control.forward(static_cast<const std::uint16_t*>(in.ptr),static_cast<std::uint16_t*>(control_out.ptr),rows);
                    candidate.forward(static_cast<const std::uint16_t*>(in.ptr),static_cast<std::uint16_t*>(candidate_out.ptr),rows);
                    std::vector<std::uint16_t> expected(rows*kHidden),actual(expected.size());
                    cuda_check(cudaMemcpy(expected.data(),control_out.ptr,expected.size()*2,cudaMemcpyDeviceToHost),"coalesced control readback");
                    cuda_check(cudaMemcpy(actual.data(),candidate_out.ptr,actual.size()*2,cudaMemcpyDeviceToHost),"coalesced candidate readback");
                    require(actual==expected,"coalesced full-layer represented output differs");
                }
                bool trace_refused=false;
                try{candidate.trace();}catch(const std::logic_error&){trace_refused=true;}
                require(trace_refused,"coalesced scratch exposed overwritten intermediates");
                candidate.set_capture_active(true);
                bool capture_refused=false;
                try{candidate.forward(static_cast<const std::uint16_t*>(in.ptr),static_cast<std::uint16_t*>(candidate_out.ptr),1);}
                catch(const std::invalid_argument&){capture_refused=true;}
                candidate.set_capture_active(false);
                require(capture_refused,"coalesced unsupported capture admitted");
                // Refusal must precede any output mutation, for both diagnostic
                // entry points, and must leave an ordinary forward usable.
                std::vector<std::uint16_t> sentinel(16*kHidden,0x3555),unchanged(sentinel.size());
                for(bool capture:{false,true}) {
                    cuda_check(cudaMemcpy(candidate_out.ptr,sentinel.data(),sentinel.size()*2,cudaMemcpyHostToDevice),
                               "coalesced refusal sentinel upload");
                    candidate.set_capture_active(capture);
                    bool refused=false;
                    try {
                        candidate.forward(static_cast<const std::uint16_t*>(in.ptr),
                            static_cast<std::uint16_t*>(candidate_out.ptr),16,0,nullptr,!capture);
                    } catch(const std::invalid_argument&) {refused=true;}
                    candidate.set_capture_active(false);
                    require(refused,"coalesced profile/capture refusal missing");
                    cuda_check(cudaMemcpy(unchanged.data(),candidate_out.ptr,unchanged.size()*2,cudaMemcpyDeviceToHost),
                               "coalesced refusal output readback");
                    require(unchanged==sentinel,"coalesced refused forward modified output");
                }
                control.forward(static_cast<const std::uint16_t*>(in.ptr),
                    static_cast<std::uint16_t*>(control_out.ptr),16);
                candidate.forward(static_cast<const std::uint16_t*>(in.ptr),
                    static_cast<std::uint16_t*>(candidate_out.ptr),16);
                std::vector<std::uint16_t> recovered(sentinel.size()),reference(sentinel.size());
                cuda_check(cudaMemcpy(recovered.data(),candidate_out.ptr,recovered.size()*2,cudaMemcpyDeviceToHost),
                           "coalesced post-refusal output readback");
                cuda_check(cudaMemcpy(reference.data(),control_out.ptr,reference.size()*2,cudaMemcpyDeviceToHost),
                           "coalesced post-refusal control readback");
                require(recovered==reference,"coalesced diagnostic refusal changed subsequent ordinary output");
                require(shared || control.workspace_bytes()-candidate.workspace_bytes()==16u*kHidden*2,
                    "coalesced private scratch accounting delta");
            }
            return 0; // Prepared parity mode excludes the historical timing loop.
        }
        Exl3FullAttentionLayer layer(weights, 14);
        if(const auto* mode=std::getenv("NINFER_TEST_ATTENTION_PAGE_BINDING");mode && std::string_view(mode)=="1") {
            // Identity/geometry-only sentinels. No forward or device access uses them.
            std::uint16_t key=0,value=0;float scores=0;
            layer.set_kv_cache(&key,&value,128);layer.set_exact_attention_scores(&scores,14);
            require(layer.supports_segmented_exact_prefix(),"page binding profile NOT_EXERCISED");
            ninfer::exl3::Exl3AttentionPageRanges original;
            original.append(&key,&value,0,64,64);
            layer.set_segmented_exact_pages(original,64);
            int rotary[3]{},other_rotary[3]{};
            layer.set_mrope_positions(rotary,7);
            layer.set_segmented_exact_pages(original,64);
            for(bool changed_pointer:{false,true}) {
                layer.set_mrope_positions(changed_pointer?other_rotary:rotary,changed_pointer?7:8);
                bool rejected=false;
                try{layer.forward(&key,&value,1,64);}
                catch(const std::invalid_argument& error) {
                    rejected=std::string(error.what())=="segmented attention rotary binding changed";
                }
                require(rejected && value==0,"changed rotary binding reached numerical forward");
                layer.set_mrope_positions(rotary,7);
            }
            for(int position:{-1,129})for(bool empty:{false,true}) {
                bool rejected=false;
                try{layer.set_segmented_exact_pages(empty?ninfer::exl3::Exl3AttentionPageRanges{}:original,position);}
                catch(const std::invalid_argument&){rejected=true;}
                const auto preserved=layer.segmented_pages_for_test();
                require(rejected && preserved.count==1 && preserved.ranges[0].k==&key &&
                    preserved.ranges[0].v==&value && preserved.ranges[0].first==0 && preserved.ranges[0].rows==64,
                    "invalid page position replaced prior binding");
            }
            layer.set_segmented_exact_pages({},128);
            require(layer.segmented_pages_for_test().count==0,"valid page binding reset failed after refusal");
            layer.set_exact_attention_scores(nullptr,0);layer.set_kv_cache(nullptr,nullptr,0);
            layer.set_mrope_positions(nullptr,0);
            const auto fixture_input=tensor_as_f16(ninfer::exl3::read_tensor(
                fixture_path,fixture,"prefill14_layer_output_input"));
            Exl3FullAttentionLayer clean(weights,14);
            DeviceBuffer input,actual,expected;
            cuda_check(cudaMalloc(&input.ptr,fixture_input.size()*2),"binding recovery input");
            cuda_check(cudaMalloc(&actual.ptr,14ULL*kHidden*2),"binding recovery output");
            cuda_check(cudaMalloc(&expected.ptr,14ULL*kHidden*2),"binding recovery control");
            cuda_check(cudaMemcpy(input.ptr,fixture_input.data(),fixture_input.size()*2,cudaMemcpyHostToDevice),
                "binding recovery fixture upload");
            clean.forward(static_cast<const std::uint16_t*>(input.ptr),static_cast<std::uint16_t*>(expected.ptr),14,0);
            layer.forward(static_cast<const std::uint16_t*>(input.ptr),static_cast<std::uint16_t*>(actual.ptr),14,0);
            std::vector<std::uint16_t> control_values(14*kHidden),recovered_values(14*kHidden);
            cuda_check(cudaMemcpy(control_values.data(),expected.ptr,control_values.size()*2,cudaMemcpyDeviceToHost),
                "binding recovery control readback");
            cuda_check(cudaMemcpy(recovered_values.data(),actual.ptr,recovered_values.size()*2,cudaMemcpyDeviceToHost),
                "binding recovery output readback");
            require(control_values==recovered_values,"rejected page/rotary binding contaminated subsequent default forward");
            const auto control_stages=stages(clean.trace(),"prefill14");
            const auto recovered_stages=stages(layer.trace(),"prefill14");
            require(control_stages.size()==recovered_stages.size(),"binding recovery trace extent");
            for(std::size_t index=0;index<control_stages.size();++index) {
                const auto& baseline=control_stages[index];const auto& recovered=recovered_stages[index];
                require(baseline.actual && recovered.actual && baseline.fixture==recovered.fixture,
                    "binding recovery missing/misaligned intermediate");
                const auto shape=tensor_as_f16(ninfer::exl3::read_tensor(fixture_path,fixture,baseline.fixture));
                std::vector<std::uint16_t> a(shape.size()),b(shape.size());
                cuda_check(cudaMemcpy(a.data(),baseline.actual,a.size()*2,cudaMemcpyDeviceToHost),"binding recovery stage control");
                cuda_check(cudaMemcpy(b.data(),recovered.actual,b.size()*2,cudaMemcpyDeviceToHost),"binding recovery stage actual");
                require(a==b,std::string("binding recovery intermediate differs: ")+baseline.label);
            }
            return 0;
        }
        const auto lifetime_children=layer.linear_workspace_owners();
        for(unsigned i=0;i<lifetime_children.size();++i) {
            require(lifetime_children[i]!=nullptr,"attention lifetime child missing");
            for(unsigned j=0;j<i;++j)require(lifetime_children[i]!=lifetime_children[j],
                "attention lifetime child duplicated across inventory slots");
            require(lifetime_children[i]->can_attach_metadata_credit(),"attention child metadata already attached");
            require(lifetime_children[i]->can_attach_device_credit(),"unreserved attention child unexpectedly received constructor device credit");
        }
        require(layer.fixed_owner_metadata_bytes()==sizeof(Exl3FullAttentionLayer)+7*Exl3CudaLinearWorkspace::metadata_bytes()+
            18*Exl3LayerBufferRetirement::record_bytes(),"attention linear and buffer retirement metadata");
        const auto buffer_children=layer.buffer_retirement_owners();
        std::uint64_t owned_bytes=0;
        for(const auto* child:lifetime_children)owned_bytes+=child->workspace_bytes();
        for(unsigned i=0;i<buffer_children.size();++i)if(const auto* child=buffer_children[i]) {
            require(child->bytes()>0,"attention retained buffer lacks extent");
            for(unsigned j=0;j<i;++j)require(child!=buffer_children[j],"attention buffer has duplicate owner");
            owned_bytes+=child->bytes();
        }
        require(owned_bytes==layer.workspace_bytes(),"attention child inventory omits or double counts owned device storage");
        require(layer.fixed_owner_metadata_bytes()==Exl3FullAttentionLayer::fixed_owner_metadata_required(),
            "attention fixed metadata requirement mismatch");
        require(layer.workspace_bytes()==14ull*(49152ull*2+59392ull*5*4+117760ull*2),
            "independent full-attention actual allocation ledger");
        std::vector<Metrics> prefill_metrics;
        const auto prefill_input = ninfer::exl3::read_tensor(fixture_path, fixture, "prefill14_layer_output_input");
        const auto prefill_values = tensor_as_f16(prefill_input);
        DeviceBuffer input_device, output_device;
        cuda_check(cudaMalloc(&input_device.ptr, prefill_values.size() * sizeof(std::uint16_t)), "allocate E3A prefill input");
        cuda_check(cudaMalloc(&output_device.ptr, 14u * kHidden * sizeof(std::uint16_t)), "allocate E3A prefill output");
        cuda_check(cudaMemcpy(input_device.ptr, prefill_values.data(), prefill_values.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice), "upload E3A prefill input");
        layer.forward(static_cast<const std::uint16_t*>(input_device.ptr), static_cast<std::uint16_t*>(output_device.ptr), 14, 0);
        cuda_check(cudaDeviceSynchronize(), "synchronize E3A prefill correctness run");
        compare_trace(layer.trace(), fixture, "prefill14", prefill_metrics);

        std::vector<Metrics> m1_metrics;
        const auto m1_input = ninfer::exl3::read_tensor(fixture_path, fixture, "m1_layer_output_input");
        const auto m1_values = tensor_as_f16(m1_input);
        cuda_check(cudaMemcpy(input_device.ptr, m1_values.data(), m1_values.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice), "upload E3A M=1 input");
        layer.forward(static_cast<const std::uint16_t*>(input_device.ptr), static_cast<std::uint16_t*>(output_device.ptr), 1, 0);
        cuda_check(cudaDeviceSynchronize(), "synchronize E3A M=1 correctness run");
        compare_trace(layer.trace(), fixture, "m1", m1_metrics);

        std::vector<Exl3FullAttentionLayerTimings> timing_runs;
        for (int i = 0; i < 5; ++i) {
            layer.forward(static_cast<const std::uint16_t*>(input_device.ptr), static_cast<std::uint16_t*>(output_device.ptr), 1, 0, nullptr, true);
            timing_runs.push_back(layer.last_timings());
        }
        const auto timings = average_timings(timing_runs);
        static constexpr const char* labels[] = {
            "input norm", "Q projection", "K projection", "V projection", "Q/K norm",
            "RoPE", "ordinary attention", "output projection", "residual/norm",
            "MLP gate", "MLP up", "activation", "MLP down/final residual"};
        std::cout << "TIMING M=1 total_us=" << timings.total_microseconds << '\n';
        for (std::size_t i = 0; i < timings.microseconds.size(); ++i) {
            std::cout << "TIMING M=1 " << labels[i] << " us=" << timings.microseconds[i] << '\n';
        }
        std::cout << "E3A native EXL3 projections=7 K={" << q.metadata.K << ',' << k.metadata.K << ',' <<
            v.metadata.K << ',' << o.metadata.K << ',' << gate.metadata.K << ',' << up.metadata.K << ',' << down.metadata.K << "}\n";
        std::cout << "E3A workspace_bytes=" << layer.workspace_bytes() << '\n';
        // The public native layer output is F16; at values around 186, one F16
        // ulp is 0.125.  The final gate therefore uses the measured F16 rounding
        // envelope plus a relative bound, while every intermediate is retained.
        require(prefill_metrics.back().max_abs <= 0.125 && prefill_metrics.back().relative_l2 <= 0.001,
                "E3A prefill final layer gate failed");
        require(m1_metrics.back().max_abs <= 0.125 && m1_metrics.back().relative_l2 <= 0.001,
                "E3A M=1 final layer gate failed");
        std::cout << "E3A full-attention layer test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "E3A full-attention layer test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
