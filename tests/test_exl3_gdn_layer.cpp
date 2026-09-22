#include "exl3/gdn_layer.h"
#include "exl3/safetensors.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
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
#include <vector>

namespace {
using namespace ninfer::exl3;
constexpr int HIDDEN = 5120, QKV = 10240, Z = 6144, HEADS = 48, INTERMEDIATE = 17408;
constexpr int PREFILL = 11, DECODE_STEPS = 4;

void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
std::string env(const char* name) { const char* v = std::getenv(name); return v ? v : ""; }
void cuda_check(cudaError_t e, const char* op) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(e));
}

struct DeviceBuffer {
    void* ptr = nullptr;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

void check_recurrent_layout_contract() {
    // These are descriptor-only contract tests: use non-overlapping synthetic
    // device extents large enough for the canonical full-layer layout. Small
    // adjacent stack objects correctly fail the production overlap guard.
    auto* recurrent_storage=reinterpret_cast<void*>(
        static_cast<std::uintptr_t>(0x10000000));
    auto* convolution_storage=reinterpret_cast<void*>(
        static_cast<std::uintptr_t>(0x20000000));
    auto model=std::make_shared<int>(1);
    auto foreign_model=std::make_shared<int>(2);
    const auto* layer_owner=reinterpret_cast<const Exl3GdnLayer*>(
        static_cast<std::uintptr_t>(1));
    Exl3GdnLayerCheckpoint checkpoint{
        recurrent_storage,Exl3GdnRecurrentLayout::recurrent_bytes,
        convolution_storage,Exl3GdnRecurrentLayout::convolution_storage_bytes};
    checkpoint.model_owner=model;
    checkpoint.model_layer=5;
    checkpoint.position=23;
    checkpoint.recurrent_layer_stride_bytes=Exl3GdnRecurrentLayout::recurrent_bytes;
    checkpoint.convolution_layer_stride_bytes=
        Exl3GdnRecurrentLayout::convolution_storage_bytes;
    checkpoint.owner=layer_owner;
    checkpoint.generation=17;
    checkpoint.backing_generation=17;
    checkpoint.generation_authority=&checkpoint.backing_generation;
    checkpoint.saved_recurrent_state_device=checkpoint.recurrent_state_device;
    checkpoint.saved_conv_state_device=checkpoint.conv_state_device;
    require(checkpoint.saved_for(model,5,23,17,layer_owner),
            "canonical recurrent layout was refused");

    auto wrong_stride=checkpoint;
    --wrong_stride.recurrent_layer_stride_bytes;
    require(!wrong_stride.saved_for(model,5,23,17,layer_owner),
            "wrong recurrent stride was accepted");
    auto truncated_recurrent=checkpoint;
    --truncated_recurrent.recurrent_state_capacity_bytes;
    require(!truncated_recurrent.saved_for(model,5,23,17,layer_owner),
            "truncated recurrent plane was accepted");
    auto truncated_convolution=checkpoint;
    --truncated_convolution.conv_state_capacity_bytes;
    require(!truncated_convolution.saved_for(model,5,23,17,layer_owner),
            "truncated convolution plane was accepted");
    auto overlapping_planes=checkpoint;
    overlapping_planes.conv_state_device=overlapping_planes.recurrent_state_device;
    overlapping_planes.saved_conv_state_device=overlapping_planes.conv_state_device;
    require(!overlapping_planes.saved_for(model,5,23,17,layer_owner),
            "overlapping recurrent checkpoint planes were accepted");
    auto mixed_layer=checkpoint;
    mixed_layer.model_layer=9;
    require(!mixed_layer.saved_for(model,5,23,17,layer_owner),
            "mixed recurrent layer was accepted");
    require(!checkpoint.saved_for(foreign_model,5,23,17,layer_owner),
            "foreign recurrent model owner was accepted");
    auto stale_generation=checkpoint;
    ++checkpoint.backing_generation;
    require(!stale_generation.saved_for(model,5,23,17,layer_owner),
            "stale recurrent generation was accepted");
    checkpoint.backing_generation=checkpoint.generation;
    auto missing_position=checkpoint;
    missing_position.position=-1;
    require(!missing_position.saved_for(model,5,23,17,layer_owner),
            "checkpoint without position was accepted");
    require(!checkpoint.saved_for(model,5,24,17,layer_owner),
            "wrong recurrent position frontier was accepted");
}

void check_recurrent_vector_access_contract() {
    constexpr std::size_t rows=8;
    constexpr std::size_t row_bytes=rows*
        Exl3GdnRecurrentLayout::value_heads*
        Exl3GdnRecurrentLayout::value_columns*sizeof(std::uint16_t);
    const auto* state=reinterpret_cast<const float*>(
        static_cast<std::uintptr_t>(0x10000000));
    const auto* values=reinterpret_cast<const std::uint16_t*>(
        static_cast<std::uintptr_t>(0x20000000));
    auto* output=reinterpret_cast<std::uint16_t*>(
        static_cast<std::uintptr_t>(0x30000000));
    Exl3GdnRecurrentVectorAccess access{
        state,Exl3GdnRecurrentLayout::recurrent_bytes,
        values,row_bytes,output,row_bytes,rows,
        Exl3GdnRecurrentLayout::value_columns};
    require(access.pair_columns_supported(),
            "aligned recurrent vector access was refused");

    auto misaligned_state=access;
    misaligned_state.state=reinterpret_cast<const float*>(
        reinterpret_cast<std::uintptr_t>(state)+sizeof(float));
    require(!misaligned_state.pair_columns_supported(),
            "misaligned FP32 state vector access was accepted");
    auto misaligned_values=access;
    misaligned_values.values=reinterpret_cast<const std::uint16_t*>(
        reinterpret_cast<std::uintptr_t>(values)+sizeof(std::uint16_t));
    require(!misaligned_values.pair_columns_supported(),
            "misaligned BF16 value vector access was accepted");
    auto tail=access;
    tail.active_value_columns=Exl3GdnRecurrentLayout::value_columns-1;
    require(!tail.pair_columns_supported(),
            "tail-column vector access was accepted");
    auto short_state=access;
    --short_state.state_capacity_bytes;
    require(!short_state.pair_columns_supported(),
            "short FP32 state vector extent was accepted");
    auto short_values=access;
    --short_values.values_capacity_bytes;
    require(!short_values.pair_columns_supported(),
            "short BF16 value vector extent was accepted");
    auto aliased_output=access;
    aliased_output.output=const_cast<std::uint16_t*>(values);
    require(!aliased_output.pair_columns_supported(),
            "aliased value/output vector access was accepted");
    auto adjacent=access;
    adjacent.output=reinterpret_cast<std::uint16_t*>(
        reinterpret_cast<std::uintptr_t>(values)+row_bytes);
    require(adjacent.pair_columns_supported(),
            "adjacent nonaliasing vector access was refused");
}

void check_recurrent_scratch_reuse_contract() {
    Exl3GdnScratchReuseContract split{
        1024,Exl3GdnScratchReuseContract::private_history_rows,1,
        Exl3GdnScratchReuseContract::wide_bytes_required(1024)};
    require(split.valid(),"canonical recurrent scratch plan was refused");
    for(const std::size_t rows:{2,4,8})
        require(split.history_storage(rows,true,true)==
                    Exl3GdnHistoryStorage::private_retained,
                "verifier/graph history escaped private storage");
    require(split.history_storage(1,false,true)==
                Exl3GdnHistoryStorage::private_retained,
            "B1 graph scratch plan was refused");
    require(split.history_storage(16,false,false)==
                Exl3GdnHistoryStorage::private_retained,
            "native16 private scratch plan was refused");
    require(split.history_storage(32,false,false)==
                Exl3GdnHistoryStorage::shared_wide,
            "wide transient scratch plan was refused");
    require(split.history_storage(3,true,false)==
                Exl3GdnHistoryStorage::private_retained &&
            !Exl3GdnScratchReuseContract::policy_horizon(3) &&
            Exl3GdnScratchReuseContract::policy_horizon(2) &&
            Exl3GdnScratchReuseContract::policy_horizon(4) &&
            Exl3GdnScratchReuseContract::policy_horizon(8),
            "exact odd-width repair was conflated with horizon policy");
    require(split.history_storage(9,true,false)==
                Exl3GdnHistoryStorage::refused,
            "unsupported retained history width was accepted");
    require(split.history_storage(32,true,false)==
                Exl3GdnHistoryStorage::refused &&
            split.history_storage(32,false,true)==
                Exl3GdnHistoryStorage::refused,
            "saved or graph history was assigned shared wide scratch");
    require(split.history_storage(1025,false,false)==
                Exl3GdnHistoryStorage::refused,
            "recurrent scratch capacity pressure was accepted");
    auto short_wide=split;
    --short_wide.shared_wide_bytes;
    require(!short_wide.valid() &&
                short_wide.history_storage(32,false,false)==
                    Exl3GdnHistoryStorage::refused,
            "undersized shared wide scratch was accepted");
    Exl3GdnScratchReuseContract private_only{16,16,0,0};
    require(private_only.valid() &&
                private_only.history_storage(8,true,false)==
                    Exl3GdnHistoryStorage::private_retained,
            "pending checkpoint incorrectly blocked private verifier scratch");
}

void check_gdn_stage_fusion_contract() {
    Exl3GdnStageFusionContract exact{8,true,true,true};
    require(exact.exact_route_supported(),
            "exact GDN stage-fusion route was refused");
    for(const auto refused:std::array<Exl3GdnStageFusionContract,5>{
            Exl3GdnStageFusionContract{1,true,true,true},
            Exl3GdnStageFusionContract{1025,true,true,true},
            Exl3GdnStageFusionContract{8,false,true,true},
            Exl3GdnStageFusionContract{8,true,false,true},
            Exl3GdnStageFusionContract{8,true,true,false}})
        require(!refused.exact_route_supported(),
                "unsupported GDN stage-fusion topology was accepted");

    Exl3GdnStageFusionFixtureView view;
    view.q=reinterpret_cast<const std::uint16_t*>(0x10000000);
    view.k=reinterpret_cast<const std::uint16_t*>(0x11000000);
    view.v=reinterpret_cast<const std::uint16_t*>(0x12000000);
    view.g=reinterpret_cast<const float*>(0x13000000);
    view.beta=reinterpret_cast<const float*>(0x14000000);
    view.canonical_state=reinterpret_cast<float*>(0x20000000);
    view.fused_state=reinterpret_cast<float*>(0x21000000);
    view.canonical_output=reinterpret_cast<std::uint16_t*>(0x30000000);
    view.fused_output=reinterpret_cast<std::uint16_t*>(0x31000000);
    view.normalized_q=reinterpret_cast<float*>(0x40000000);
    view.normalized_k=reinterpret_cast<float*>(0x41000000);
    view.alpha=reinterpret_cast<float*>(0x42000000);
    view.rows=8;
    view.qk_capacity_bytes=8u*16u*128u*sizeof(std::uint16_t);
    view.value_capacity_bytes=8u*48u*128u*sizeof(std::uint16_t);
    view.control_capacity_bytes=8u*48u*sizeof(float);
    view.state_capacity_bytes=Exl3GdnRecurrentLayout::recurrent_bytes;
    view.normalized_capacity_bytes=8u*16u*128u*sizeof(float);
    require(view.valid(),"canonical GDN stage-fusion fixture was refused");
    auto short_qk=view;--short_qk.qk_capacity_bytes;
    require(!short_qk.valid(),"short GDN fusion Q/K extent was accepted");
    auto shared_state=view;shared_state.fused_state=shared_state.canonical_state;
    require(!shared_state.valid(),"aliased GDN fusion state was accepted");
}

void check_gdn_continuation_history_contract() {
    auto storage=std::make_shared<int>(1),model=std::make_shared<int>(2);
    std::uint64_t generation=7;
    Exl3GdnContinuationHistoryView view{
        storage,model,&generation,generation,5,320,8,2,3,
        reinterpret_cast<const std::uint16_t*>(0x10000000),
        reinterpret_cast<const std::uint16_t*>(0x11000000),
        reinterpret_cast<const std::uint16_t*>(0x12000000),
        reinterpret_cast<const std::uint16_t*>(0x13000000),
        reinterpret_cast<const float*>(0x14000000),
        reinterpret_cast<const float*>(0x15000000),
        reinterpret_cast<const std::uint16_t*>(0x16000000)};
    require(view.current(),"current owning GDN history view was refused");
    auto missing_row=view;missing_row.first_row=8;
    require(!missing_row.current(),"missing GDN history row was accepted");
    auto wrong_layer=view;wrong_layer.model_layer=3;
    require(!wrong_layer.current(),"full-attention layer accepted GDN history");
    auto no_owner=view;no_owner.storage_owner.reset();
    require(!no_owner.current(),"unowned GDN history view was accepted");
    ++generation;
    require(!view.current(),"stale GDN history generation was accepted");
}

std::unique_ptr<DeviceBuffer> upload(std::span<const std::byte> bytes, const char* label) {
    auto result = std::make_unique<DeviceBuffer>();
    cuda_check(cudaMalloc(&result->ptr, bytes.size_bytes()), label);
    try { cuda_check(cudaMemcpy(result->ptr, bytes.data(), bytes.size_bytes(), cudaMemcpyHostToDevice), "upload E3B tensor"); }
    catch (...) { cudaFree(result->ptr); result->ptr = nullptr; throw; }
    return result;
}

std::unique_ptr<DeviceBuffer> allocate_device(std::size_t bytes,const char* label) {
    auto result=std::make_unique<DeviceBuffer>();
    cuda_check(cudaMalloc(&result->ptr,bytes),label);
    return result;
}

TensorPayload load_tensor(const IndexedSafetensors& collection, const std::string& name) {
    for (const auto& shard : collection.shards) if (shard.header.find(name)) return read_tensor(shard.path, shard.header, name);
    throw std::runtime_error("indexed tensor is absent: " + name);
}

float bf16_float(std::uint16_t bits) {
    std::uint32_t u = static_cast<std::uint32_t>(bits) << 16u; float x; std::memcpy(&x, &u, sizeof(x)); return x;
}
float half_float(std::uint16_t bits) {
    const std::uint32_t sign = (bits & 0x8000u) << 16u, exp = (bits >> 10u) & 31u, frac = bits & 1023u;
    std::uint32_t u = sign;
    if (exp == 0) {
        if (frac) { std::uint32_t m = frac; int shift = 0; while (!(m & 1024u)) { m <<= 1u; ++shift; }
            u |= static_cast<std::uint32_t>(127 - 14 - shift) << 23u; u |= (m & 1023u) << 13u; }
    } else if (exp == 31) u |= 0x7f800000u | (frac << 13u);
    else u |= (exp + 112u) << 23u | (frac << 13u);
    float x; std::memcpy(&x, &u, sizeof(x)); return x;
}
std::uint16_t half_bits(float x) {
    std::uint32_t bits; std::memcpy(&bits, &x, sizeof(bits));
    const std::uint32_t sign = (bits >> 16u) & 0x8000u, exp = (bits >> 23u) & 255u, frac = bits & 0x7fffffu;
    if (exp == 255u) return static_cast<std::uint16_t>(sign | (frac ? 0x7e00u : 0x7c00u));
    const int e = static_cast<int>(exp) - 127;
    if (e > 15) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (e >= -14) { std::uint32_t he = static_cast<std::uint32_t>(e + 15), hf = frac >> 13u, rem = frac & 8191u;
        if (rem > 4096u || (rem == 4096u && (hf & 1u))) { if (++hf == 1024u) { hf = 0; ++he; } }
        return he >= 31u ? static_cast<std::uint16_t>(sign | 0x7c00u) : static_cast<std::uint16_t>(sign | (he << 10u) | hf); }
    if (e < -25) return static_cast<std::uint16_t>(sign);
    const std::uint32_t m = frac | 0x800000u; const int shift = -e - 14;
    std::uint32_t hf = m >> (shift + 13), rem = m & ((1u << (shift + 13)) - 1u), mid = 1u << (shift + 12);
    if (rem > mid || (rem == mid && (hf & 1u))) ++hf;
    return static_cast<std::uint16_t>(sign | hf);
}

std::vector<float> as_float(const TensorPayload& t) {
    std::vector<float> out;
    if (t.info.dtype == "F32") { auto x = t.typed<float>("F32"); out.assign(x.begin(), x.end()); }
    else if (t.info.dtype == "F16") for (auto x : t.typed<std::uint16_t>("F16")) out.push_back(half_float(x));
    else if (t.info.dtype == "BF16") for (auto x : t.typed<std::uint16_t>("BF16")) out.push_back(bf16_float(x));
    else throw std::runtime_error(t.info.name + " has unsupported fixture dtype " + t.info.dtype);
    return out;
}

struct Linear {
    Exl3CudaLinearWeights view{};
    Exl3CudaLinearMetadata metadata{};
    std::vector<TensorPayload> host;
};
Linear make_linear(const IndexedSafetensors& c, const std::string& prefix, int in, int out,
                   std::vector<std::unique_ptr<DeviceBuffer>>& device) {
    Linear r;
    auto get = [&](const char* suffix) { r.host.push_back(load_tensor(c, prefix + "." + suffix)); return r.host.back(); };
    auto trellis = get("trellis"), suh = get("suh"), svh = get("svh"), mul1 = get("mul1");
    require(trellis.info.dtype == "I16" && trellis.info.shape.size() == 3, prefix + " trellis metadata mismatch");
    require(suh.info.dtype == "F16" && svh.info.dtype == "F16" && mul1.info.dtype == "I32", prefix + " auxiliary metadata mismatch");
    auto add = [&](const TensorPayload& t, const char* label) { device.push_back(upload(t.bytes(), label)); return device.back()->ptr; };
    r.view.trellis = static_cast<const std::uint16_t*>(add(trellis, "upload E3B trellis"));
    r.view.suh = static_cast<const std::uint16_t*>(add(suh, "upload E3B suh"));
    r.view.svh = static_cast<const std::uint16_t*>(add(svh, "upload E3B svh"));
    r.view.mul1 = static_cast<const std::int32_t*>(add(mul1, "upload E3B mul1"));
    require(trellis.info.shape[0] == static_cast<std::uint64_t>(in / 16) && trellis.info.shape[1] == static_cast<std::uint64_t>(out / 16), prefix + " logical shape mismatch");
    r.metadata = {in, out, static_cast<int>(trellis.info.shape[2] / 16), false, true, false};
    return r;
}

const std::uint16_t* load_device_raw(const TensorPayload& t, std::vector<std::unique_ptr<DeviceBuffer>>& device, const char* label) {
    require(t.info.dtype == "F16" || t.info.dtype == "BF16", t.info.name + " must be F16/BF16");
    return static_cast<const std::uint16_t*>(device.emplace_back(upload(t.bytes(), label))->ptr);
}
const std::uint16_t* load_device_f16(const TensorPayload& t, std::vector<std::unique_ptr<DeviceBuffer>>& device, const char* label) {
    std::vector<std::uint16_t> values;
    if (t.info.dtype == "F16") { auto x = t.typed<std::uint16_t>("F16"); values.assign(x.begin(), x.end()); }
    else { require(t.info.dtype == "BF16", t.info.name + " must be BF16/F16"); for (auto x : t.typed<std::uint16_t>("BF16")) values.push_back(half_bits(bf16_float(x))); }
    auto bytes = std::span<const std::byte>(reinterpret_cast<const std::byte*>(values.data()), values.size() * sizeof(std::uint16_t));
    return static_cast<const std::uint16_t*>(device.emplace_back(upload(bytes, label))->ptr);
}
const float* load_device_f32(const TensorPayload& t, std::vector<std::unique_ptr<DeviceBuffer>>& device, const char* label) {
    auto values = as_float(t); auto bytes = std::span<const std::byte>(reinterpret_cast<const std::byte*>(values.data()), values.size() * sizeof(float));
    return static_cast<const float*>(device.emplace_back(upload(bytes, label))->ptr);
}

struct Metric { double max_abs = 0, mean_abs = 0, rms = 0, rel = 0, p99 = 0, p999 = 0; };
Metric compare(const std::vector<float>& actual, const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "E3B stage size mismatch");
    Metric m; std::vector<double> errors; errors.reserve(actual.size()); double sum = 0, ss = 0, es = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) { double e = std::abs(actual[i] - expected[i]); errors.push_back(e); sum += e; ss += e * e; es += static_cast<double>(expected[i]) * expected[i]; m.max_abs = std::max(m.max_abs, e); }
    std::sort(errors.begin(), errors.end()); auto pct = [&](double p) { return errors[std::min(errors.size() - 1, static_cast<std::size_t>(p * (errors.size() - 1)))]; };
    m.mean_abs = sum / actual.size(); m.rms = std::sqrt(ss / actual.size()); m.rel = es == 0 ? std::sqrt(ss) : std::sqrt(ss / es); m.p99 = pct(.99); m.p999 = pct(.999); return m;
}

enum class Kind { F16, BF16, F32 };
struct Stage { const char* label; const char* suffix; const void* actual; Kind kind; };

std::vector<Stage> stage_list(const Exl3GdnLayerTrace& t) {
    return {{"layer input", "layer_output_input", t.layer_input, Kind::F16}, {"input norm", "input_norm", t.input_norm, Kind::F16},
        {"qkv projection", "qkv_projection", t.qkv_projection, Kind::F16}, {"z projection", "z_projection", t.z_projection, Kind::F16},
        {"b projection", "b_projection", t.b_projection, Kind::F32}, {"a projection", "a_projection", t.a_projection, Kind::F32},
        {"conv input", "conv_input", t.conv_input, Kind::BF16}, {"conv output", "conv_output", t.conv_output, Kind::BF16},
        {"beta", "beta", t.beta, Kind::F32}, {"g", "g", t.g, Kind::F32}, {"state before", "state_before", t.state_before, Kind::F32},
        {"state after", "state_after", t.state_after, Kind::F32}, {"linear attention output", "linear_attention_output", t.linear_attention_output, Kind::BF16},
        {"GDN norm input", "gdn_norm_input", t.gdn_norm_input, Kind::BF16}, {"GDN norm", "gdn_norm", t.gdn_norm, Kind::F16},
        {"output projection input", "output_projection_input", t.output_projection_input, Kind::F16}, {"output projection", "output_projection", t.output_projection, Kind::F16},
        {"post-attention residual", "post_attention_residual", t.post_attention_residual, Kind::F16},
        {"MLP input", "mlp_norm", t.mlp_input, Kind::F16},
        {"gate projection", "gate_projection", t.gate_projection, Kind::F16}, {"up projection", "up_projection", t.up_projection, Kind::F16},
        {"activated MLP", "activated_mlp", t.activated_mlp, Kind::F16}, {"down projection", "down_projection", t.down_projection, Kind::F16},
        {"layer output", "layer_output", t.layer_output, Kind::F16}};
}

Metric compare_stage(const Exl3GdnLayerTrace& trace, const SafetensorsHeader& fixture, const std::filesystem::path& path,
                     const std::string& tag, const Stage& stage) {
    const std::string name = tag + "_" + stage.suffix; const auto* info = fixture.find(name); require(info, "fixture stage missing: " + name);
    auto expected = as_float(read_tensor(path, fixture, name)); const std::size_t n = expected.size(); std::vector<float> actual(n);
    if (stage.kind == Kind::F32) {
        cuda_check(cudaMemcpy(actual.data(), stage.actual, n * sizeof(float), cudaMemcpyDeviceToHost), "download E3B FP32 stage");
    }
    else { std::vector<std::uint16_t> bits(n); cuda_check(cudaMemcpy(bits.data(), stage.actual, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "download E3B stage"); for (std::size_t i = 0; i < n; ++i) actual[i] = stage.kind == Kind::F16 ? half_float(bits[i]) : bf16_float(bits[i]); }
    const auto m = compare(actual, expected);
    std::cout << "STAGE " << tag << " " << stage.label << " max_abs=" << std::setprecision(10) << m.max_abs << " mean_abs=" << m.mean_abs << " rms=" << m.rms << " relative_l2=" << m.rel << " p99=" << m.p99 << " p999=" << m.p999 << '\n';
    return m;
}

std::vector<Metric> compare_trace(const Exl3GdnLayerTrace& trace, const SafetensorsHeader& fixture, const std::filesystem::path& path, const std::string& tag) {
    std::vector<Metric> result; for (const auto& stage : stage_list(trace)) result.push_back(compare_stage(trace, fixture, path, tag, stage)); return result;
}

void compare_conv_state(const Exl3GdnLayer& layer, const SafetensorsHeader& fixture, const std::filesystem::path& path, const std::string& tag) {
    const std::string name = tag + "_conv_state_after"; require(fixture.find(name), "fixture stage missing: " + name); auto expected = as_float(read_tensor(path, fixture, name));
    std::vector<std::uint16_t> bits(expected.size()); cuda_check(cudaMemcpy(bits.data(), layer.conv_state_device(), bits.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "download E3B convolution state"); std::vector<float> actual(bits.size()); for (std::size_t i = 0; i < bits.size(); ++i) actual[i] = bf16_float(bits[i]);
    const auto m = compare(actual, expected);
    std::cout << "STATE " << tag << " convolution max_abs=" << m.max_abs << " relative_l2=" << m.rel << '\n';
    require(m.max_abs <= .03125 && m.rel <= .006, "E3B convolution state numerical gate failed for " + tag);
}

std::vector<float> download_state(const Exl3GdnLayer& layer) {
    std::vector<float> state(layer.recurrent_state_bytes() / sizeof(float)); cuda_check(cudaMemcpy(state.data(), layer.recurrent_state_device(), layer.recurrent_state_bytes(), cudaMemcpyDeviceToHost), "download E3B recurrent state"); return state;
}

std::vector<std::byte> download_bytes(const void* device, std::size_t bytes, const char* label) {
    std::vector<std::byte> result(bytes);
    cuda_check(cudaMemcpy(result.data(), device, bytes, cudaMemcpyDeviceToHost), label);
    return result;
}

bool any_nonzero(const std::vector<std::byte>& bytes) {
    return std::any_of(bytes.begin(), bytes.end(), [](std::byte value) { return value != std::byte{0}; });
}

void run_gdn_stage_fusion_fixtures() {
    constexpr int rows=8,key_heads=16,key_columns=128,value_heads=48;
    constexpr std::size_t qk_elements=rows*key_heads*key_columns;
    constexpr std::size_t value_elements=rows*value_heads*key_columns;
    constexpr std::size_t control_elements=rows*value_heads;
    constexpr std::array<std::uint16_t,8> represented{
        0x0000,0x3a80,0x3f00,0xbf00,0x3f80,0xbf80,0x4180,0xc180};
    std::vector<std::uint16_t> q(qk_elements),k(qk_elements),v(value_elements);
    std::vector<float> g(control_elements),beta(control_elements);
    std::vector<float> initial_state(
        Exl3GdnRecurrentLayout::recurrent_elements,0.0009765625f);
    for(int row=0;row<rows;++row) {
        for(int qh=0;qh<key_heads;++qh) {
            const auto base=(row*key_heads+qh)*key_columns;
            if(qh%4==1) {
                q[base]=0x3f80;k[base]=0x3f80;
            } else for(int column=0;column<key_columns;++column) {
                q[base+column]=represented[(column+qh+row)%represented.size()];
                k[base+column]=represented[(column*3+qh+row)%represented.size()];
            }
        }
        for(int head=0;head<value_heads;++head) {
            const int mode=head%4;
            g[row*value_heads+head]=mode==0?2.0f:
                (mode==2?-16.0f:(mode==3?-0.001f:0.0f));
            beta[row*value_heads+head]=mode==0?1.0f:
                (mode==2?0.0f:(mode==3?0.5f:1.0f));
            for(int column=0;column<key_columns;++column)
                v[(row*value_heads+head)*key_columns+column]=
                    represented[2+(column+head+row)%6];
        }
    }
    // The first row of every mode-1 head starts with K concentrated in column
    // zero and state almost equal to V/K. This makes beta*(V-KV) a small
    // subtraction while later rows expose cumulative drift.
    for(int head=1;head<value_heads;head+=4)
        for(int value_column=0;value_column<key_columns;++value_column) {
            const auto value=bf16_float(v[head*key_columns+value_column]);
            initial_state[static_cast<std::size_t>(head)*key_columns*key_columns+
                          value_column]=value*1.0000005f;
        }

    auto q_device=upload({reinterpret_cast<const std::byte*>(q.data()),
                           q.size()*sizeof(q[0])},"allocate fusion Q");
    auto k_device=upload({reinterpret_cast<const std::byte*>(k.data()),
                           k.size()*sizeof(k[0])},"allocate fusion K");
    auto v_device=upload({reinterpret_cast<const std::byte*>(v.data()),
                           v.size()*sizeof(v[0])},"allocate fusion V");
    auto g_device=upload({reinterpret_cast<const std::byte*>(g.data()),
                           g.size()*sizeof(g[0])},"allocate fusion g");
    auto beta_device=upload({reinterpret_cast<const std::byte*>(beta.data()),
                              beta.size()*sizeof(beta[0])},"allocate fusion beta");
    auto canonical_state=upload({
        reinterpret_cast<const std::byte*>(initial_state.data()),
        initial_state.size()*sizeof(initial_state[0])},"allocate canonical state");
    auto fused_state=upload({reinterpret_cast<const std::byte*>(initial_state.data()),
                             initial_state.size()*sizeof(initial_state[0])},
                            "allocate fused state");
    auto canonical_output=allocate_device(
        value_elements*sizeof(std::uint16_t),"allocate canonical output");
    auto fused_output=allocate_device(
        value_elements*sizeof(std::uint16_t),"allocate fused output");
    auto normalized_q=allocate_device(qk_elements*sizeof(float),
                                      "allocate normalized Q");
    auto normalized_k=allocate_device(qk_elements*sizeof(float),
                                      "allocate normalized K");
    auto alpha=allocate_device(control_elements*sizeof(float),
                               "allocate fusion alpha");
    Exl3GdnStageFusionFixtureView view{
        static_cast<const std::uint16_t*>(q_device->ptr),
        static_cast<const std::uint16_t*>(k_device->ptr),
        static_cast<const std::uint16_t*>(v_device->ptr),
        static_cast<const float*>(g_device->ptr),
        static_cast<const float*>(beta_device->ptr),
        static_cast<float*>(canonical_state->ptr),
        static_cast<float*>(fused_state->ptr),
        static_cast<std::uint16_t*>(canonical_output->ptr),
        static_cast<std::uint16_t*>(fused_output->ptr),
        static_cast<float*>(normalized_q->ptr),
        static_cast<float*>(normalized_k->ptr),static_cast<float*>(alpha->ptr),
        q.size()*sizeof(q[0]),v.size()*sizeof(v[0]),g.size()*sizeof(g[0]),
        initial_state.size()*sizeof(initial_state[0]),
        q.size()*sizeof(float),rows};
    exl3_gdn_stage_fusion_fixture(view);
    cuda_check(cudaDeviceSynchronize(),"synchronize GDN stage-fusion fixture");
    require(download_bytes(canonical_state->ptr,
                initial_state.size()*sizeof(float),"download canonical fusion state")==
            download_bytes(fused_state->ptr,
                initial_state.size()*sizeof(float),"download fused fusion state"),
            "GDN fusion recurrent drift differs from canonical FP32 state");
    require(download_bytes(canonical_output->ptr,
                value_elements*sizeof(std::uint16_t),"download canonical fusion output")==
            download_bytes(fused_output->ptr,
                value_elements*sizeof(std::uint16_t),"download fused fusion output"),
            "GDN fusion BF16 output differs at the original cast boundary");
    std::vector<float> q_normalized(qk_elements),k_normalized(qk_elements),
        alpha_host(control_elements);
    cuda_check(cudaMemcpy(q_normalized.data(),normalized_q->ptr,
        q_normalized.size()*sizeof(float),cudaMemcpyDeviceToHost),
        "download normalized fusion Q");
    cuda_check(cudaMemcpy(k_normalized.data(),normalized_k->ptr,
        k_normalized.size()*sizeof(float),cudaMemcpyDeviceToHost),
        "download normalized fusion K");
    cuda_check(cudaMemcpy(alpha_host.data(),alpha->ptr,
        alpha_host.size()*sizeof(float),cudaMemcpyDeviceToHost),
        "download fusion alpha");
    require(std::all_of(q_normalized.begin(),q_normalized.end(),
                [](float value){return std::isfinite(value);}) &&
            std::all_of(k_normalized.begin(),k_normalized.end(),
                [](float value){return std::isfinite(value);}) &&
            std::all_of(alpha_host.begin(),alpha_host.end(),
                [](float value){return std::isfinite(value)&&value>0.0f;}),
            "GDN fusion intermediate activation is not finite");
    for(int row=0;row<rows;++row)for(int qh=0;qh<key_heads;++qh) {
        float q_square=0.0f,k_square=0.0f;
        const auto base=(row*key_heads+qh)*key_columns;
        for(int column=0;column<key_columns;++column) {
            q_square+=q_normalized[base+column]*q_normalized[base+column];
            k_square+=k_normalized[base+column]*k_normalized[base+column];
        }
        require(q_square>0.999f&&q_square<=1.001f&&
                    k_square>0.999f&&k_square<=1.001f,
                "GDN fusion normalized intermediate lost RMS extent");
    }
}

#include "test_exl3_gdn_retained_prefix.h"
#include "test_exl3_gdn_future_oracle.h"

} // namespace

int main() {
    try {
        check_recurrent_layout_contract();
        check_recurrent_vector_access_contract();
        check_recurrent_scratch_reuse_contract();
        check_gdn_stage_fusion_contract();
        check_gdn_continuation_history_contract();
        const auto target = env("NINFER_EXL3_TARGET_PATH"), fixture_path = env("NINFER_EXL3_ORACLE_PATH");
        if (target.empty() || fixture_path.empty()) { std::cerr << "E3B skipped: set NINFER_EXL3_TARGET_PATH and NINFER_EXL3_ORACLE_PATH\n"; return 77; }
        run_gdn_stage_fusion_fixtures();
        run_gdn_known_oracle_cases();
        if(const auto future=env("NINFER_EXL3_RECURRENT_FUTURE_ORACLE_PATH");!future.empty())
            run_gdn_external_future_oracle(future);
        const auto collection = inspect_indexed_directory(target);
        const auto fixture = inspect_file(fixture_path);
        std::vector<std::unique_ptr<DeviceBuffer>> device;
        const std::string base = "model.language_model.layers.5.";
        auto qkv = make_linear(collection, base + "linear_attn.in_proj_qkv", HIDDEN, QKV, device);
        auto z = make_linear(collection, base + "linear_attn.in_proj_z", HIDDEN, Z, device);
        auto o = make_linear(collection, base + "linear_attn.out_proj", Z, HIDDEN, device);
        auto gate = make_linear(collection, base + "mlp.gate_proj", HIDDEN, INTERMEDIATE, device);
        auto up = make_linear(collection, base + "mlp.up_proj", HIDDEN, INTERMEDIATE, device);
        auto down = make_linear(collection, base + "mlp.down_proj", INTERMEDIATE, HIDDEN, device);
        auto raw = [&](const std::string& name, const char* label) { return load_tensor(collection, name); };
        auto norm = [&](const std::string& name) { return load_device_raw(raw(name, ""), device, "upload E3B raw parameter"); };
        auto f16_norm = [&](const std::string& name) { return load_device_f16(raw(name, ""), device, "upload E3B F16 norm"); };
        auto f32 = [&](const std::string& name) { return load_device_f32(raw(name, ""), device, "upload E3B FP32 parameter"); };
        Exl3GdnLayerWeights w; w.qkv=qkv.view; w.z=z.view; w.o=o.view; w.gate=gate.view; w.up=up.view; w.down=down.view;
        w.qkv_metadata=qkv.metadata; w.z_metadata=z.metadata; w.o_metadata=o.metadata; w.gate_metadata=gate.metadata; w.up_metadata=up.metadata; w.down_metadata=down.metadata;
        w.input_norm=f16_norm(base+"input_layernorm.weight"); w.gdn_norm=norm(base+"linear_attn.norm.weight"); w.post_attention_norm=f16_norm(base+"post_attention_layernorm.weight");
        w.conv_weight=norm(base+"linear_attn.conv1d.weight"); w.a_weight=norm(base+"linear_attn.in_proj_a.weight"); w.b_weight=norm(base+"linear_attn.in_proj_b.weight");
        w.a_log=f32(base+"linear_attn.A_log"); w.dt_bias=f32(base+"linear_attn.dt_bias");
        require(qkv.metadata.K == 5 && z.metadata.K == 6 && o.metadata.K == 6 && gate.metadata.K == 5 && up.metadata.K == 5 && down.metadata.K == 6, "unexpected E3B EXL3 K family");
        auto recurrent_model_owner=std::make_shared<int>(1);
        auto immutable_coefficients=
            std::make_shared<const Exl3GdnImmutableCoefficients>(
                Exl3GdnImmutableCoefficients{recurrent_model_owner,5,w});
        require(immutable_coefficients->matches(recurrent_model_owner,5,w),
                "canonical immutable GDN coefficient binding mismatch");
        auto mismatched_representation=w;
        ++mismatched_representation.qkv_metadata.K;
        require(!immutable_coefficients->matches(
                    recurrent_model_owner,5,mismatched_representation),
                "mismatched GDN coefficient representation was shared");
        require(!immutable_coefficients->matches(
                    std::make_shared<int>(2),5,w),
                "reloaded model shared old GDN coefficients");
        require(!immutable_coefficients->matches(recurrent_model_owner,9,w),
                "different GDN layer shared coefficients");
        bool coefficient_mismatch_refused=false;
        try {
            auto mismatched_owner=
                std::make_shared<const Exl3GdnImmutableCoefficients>(
                    Exl3GdnImmutableCoefficients{
                        recurrent_model_owner,5,mismatched_representation});
            Exl3GdnLayer mismatched_layer(w,PREFILL,{},{},{},{},nullptr,
                                          nullptr,0,mismatched_owner);
        } catch(const std::invalid_argument&) {
            coefficient_mismatch_refused=true;
        }
        require(coefficient_mismatch_refused,
                "GDN layer accepted mismatched immutable coefficients");
        auto retiring_model=std::make_shared<int>(3);
        std::weak_ptr<int> retired_model_witness=retiring_model;
        auto retirement_binding=
            std::make_shared<const Exl3GdnImmutableCoefficients>(
                Exl3GdnImmutableCoefficients{retiring_model,5,w});
        retiring_model.reset();
        require(!retired_model_witness.expired(),
                "immutable coefficients did not retain model backing");
        retirement_binding.reset();
        require(retired_model_witness.expired(),
                "immutable coefficient model backing did not retire");
        Exl3GdnLayer layer(w,PREFILL,{},{},{},{},nullptr,nullptr,0,
                           immutable_coefficients);
        layer.bind_recurrent_layout(recurrent_model_owner,5);
        const auto lifetime_children=layer.linear_workspace_owners();
        for(unsigned i=0;i<lifetime_children.size();++i) {
            require(lifetime_children[i]!=nullptr,"GDN lifetime child missing");
            for(unsigned j=0;j<i;++j)require(lifetime_children[i]!=lifetime_children[j],
                "GDN lifetime child duplicated across inventory slots");
            require(lifetime_children[i]->can_attach_metadata_credit(),"GDN child metadata already attached");
            require(lifetime_children[i]->can_attach_device_credit(),"unreserved GDN child unexpectedly received constructor device credit");
        }
        require(layer.fixed_owner_metadata_bytes()==sizeof(Exl3GdnLayer)+
            6*Exl3CudaLinearWorkspace::metadata_bytes()+sizeof(ninfer::DeviceArena)+34*Exl3LayerBufferRetirement::record_bytes(),
            "GDN linear, buffer retirement records and arena");
        const auto buffer_children=layer.buffer_retirement_owners();
        std::uint64_t child_bytes=0;
        for(const auto* child:lifetime_children)child_bytes+=child->workspace_bytes();
        for(unsigned i=0;i<buffer_children.size();++i)if(const auto* child=buffer_children[i]) {
            require(child->bytes()>0,"GDN buffer owner lacks extent");
            for(unsigned j=0;j<i;++j)require(child!=buffer_children[j],"GDN buffer owner duplicated");
            child_bytes+=child->bytes();
        }
        require(child_bytes==layer.workspace_bytes(),"GDN child inventory omits storage including operation arena");
        require(layer.fixed_owner_metadata_bytes()==Exl3GdnLayer::fixed_owner_metadata_required(),
            "GDN fixed metadata requirement mismatch");
        // Independent physical ledger for this fully private layer: six linear
        // transforms plus five FP32 output splits, 22 FP16 planes, six controls,
        // two recurrent states, physical convolution/history, and operation arena.
        // Do not call the production requirement helper here.
        const std::size_t linear_bytes=PREFILL*(44032ull*2+61440ull*5*4);
        const std::size_t half_bytes=PREFILL*(6ull*5120+3ull*10240+7ull*6144+2ull*2048+3ull*17408)*2;
        const std::size_t control_bytes=PREFILL*6ull*48*4;
        const std::size_t normalized_bytes=env("NINFER_EXL3_PREFILL_GDN_RESIDENT")=="1"
            ? PREFILL*(2ull*16*128+48)*4 : 0;
        const std::size_t state_bytes=2ull*48*128*128*4+10240ull*(4+3)*2;
        const std::size_t expected_workspace=linear_bytes+half_bytes+control_bytes+
            normalized_bytes+state_bytes+(1ull<<20);
        require(layer.workspace_bytes()==expected_workspace,"GDN independent complete workspace ledger");
        require(Exl3GdnLayer::workspace_bytes_required(PREFILL,false,false,false,false,
            env("NINFER_EXL3_PREFILL_GDN_RESIDENT")=="1")==expected_workspace,
            "GDN independent configuration requirement ledger");
        for(int rows:{-1,0,1025}) {
            bool refused=false;
            try{(void)Exl3GdnLayer::workspace_bytes_required(rows,false,false,false,false,false);}
            catch(const std::invalid_argument&){refused=true;}
            require(refused,"GDN requirement accepted unsupported rows");
        }
        for(int rows:{1,11,16,17,128,1024})for(bool accum:{false,true})
            for(bool transform:{false,true})for(bool scratch:{false,true})
                for(bool split:{false,true})for(bool resident:{false,true}) {
                    const auto linear_per_row=(transform?0ull:44032ull*2)+(accum?0ull:61440ull*5*4);
                    // Five retained planes contain 26624 features; the other
                    // seventeen contain 134144. Split-wide clips only retention.
                    const auto retained_rows=split?std::min(rows,16):rows;
                    const auto planes=retained_rows*26624ull*2+
                        (scratch?0ull:rows*134144ull*2);
                    const auto controls=rows*(scratch?2ull:6ull)*48*4;
                    const auto normalized=resident && !scratch?rows*(2ull*16*128+48)*4:0ull;
                    const auto expected=rows*linear_per_row+planes+controls+normalized+state_bytes+(1ull<<20);
                    require(Exl3GdnLayer::workspace_bytes_required(rows,accum,transform,scratch,split,resident)==expected,
                        "independent GDN borrowing and split-wide ledger");
                }
        DeviceBuffer input, output; cuda_check(cudaMalloc(&input.ptr, PREFILL * HIDDEN * 2), "allocate E3B input"); cuda_check(cudaMalloc(&output.ptr, PREFILL * HIDDEN * 2), "allocate E3B output");
        auto run = [&](const std::string& tag, int rows) { auto values = as_float(read_tensor(fixture_path, fixture, tag+"_layer_output_input")); std::vector<std::uint16_t> half(values.size()); for (std::size_t i=0;i<values.size();++i) half[i]=half_bits(values[i]); cuda_check(cudaMemcpy(input.ptr, half.data(), half.size()*2, cudaMemcpyHostToDevice), "upload E3B activation"); layer.forward(static_cast<const std::uint16_t*>(input.ptr), static_cast<std::uint16_t*>(output.ptr), rows); cuda_check(cudaDeviceSynchronize(), "synchronize E3B forward"); return compare_trace(layer.trace(), fixture, fixture_path, tag); };
        DeviceBuffer checkpoint_recurrent, checkpoint_conv;
        cuda_check(cudaMalloc(&checkpoint_recurrent.ptr, layer.recurrent_state_bytes()), "allocate E3B recurrent checkpoint");
        cuda_check(cudaMalloc(&checkpoint_conv.ptr, layer.physical_conv_state_bytes()), "allocate E3B convolution checkpoint");
        Exl3GdnLayerCheckpoint checkpoint{
            checkpoint_recurrent.ptr, layer.recurrent_state_bytes(),
            checkpoint_conv.ptr, layer.physical_conv_state_bytes()};
        checkpoint.model_owner=recurrent_model_owner;
        checkpoint.model_layer=5;
        checkpoint.recurrent_layer_stride_bytes=Exl3GdnRecurrentLayout::recurrent_bytes;
        checkpoint.convolution_layer_stride_bytes=
            Exl3GdnRecurrentLayout::convolution_storage_bytes;
        layer.reset(); auto prefill_metrics = run("prefill11", PREFILL); compare_conv_state(layer, fixture, fixture_path, "prefill11");
        int checkpoint_position=PREFILL;
        layer.save_checkpoint(checkpoint,checkpoint_position);
        cuda_check(cudaDeviceSynchronize(), "synchronize E3B transaction snapshot");
        const auto snapshot_recurrent = download_bytes(checkpoint_recurrent.ptr, layer.recurrent_state_bytes(), "download E3B recurrent checkpoint");
        const auto snapshot_conv = download_bytes(checkpoint_conv.ptr, layer.physical_conv_state_bytes(), "download E3B physical convolution checkpoint");
        require(any_nonzero(snapshot_recurrent), "E3B transaction recurrent snapshot is zero");
        require(any_nonzero(snapshot_conv), "E3B transaction convolution snapshot is zero");
        std::vector<std::vector<float>> native_states; native_states.push_back(download_state(layer));
        std::vector<std::vector<std::byte>> reference_outputs, reference_recurrent, reference_conv;
        double worst_state_rel = 0, worst_final_rel = prefill_metrics.back().rel;
        for (int step=0; step<DECODE_STEPS; ++step) {
            auto metrics = run("decode"+std::to_string(step), 1); compare_conv_state(layer, fixture, fixture_path, "decode"+std::to_string(step));
            native_states.push_back(download_state(layer));
            reference_outputs.push_back(download_bytes(output.ptr, HIDDEN * sizeof(std::uint16_t), "download E3B reference output"));
            layer.save_checkpoint(checkpoint,++checkpoint_position); cuda_check(cudaDeviceSynchronize(), "synchronize E3B reference checkpoint");
            reference_recurrent.push_back(download_bytes(checkpoint_recurrent.ptr, layer.recurrent_state_bytes(), "download E3B reference recurrent state"));
            reference_conv.push_back(download_bytes(checkpoint_conv.ptr, layer.physical_conv_state_bytes(), "download E3B reference physical convolution state"));
            worst_state_rel = std::max(worst_state_rel, metrics[11].rel); worst_final_rel = std::max(worst_final_rel, metrics.back().rel);
        }
        // Recreate the prefill snapshot, execute rejected speculative work, then roll
        // back and require exact GPU-state/output parity with uninterrupted decode.
        layer.reset(); run("prefill11", PREFILL); checkpoint_position=PREFILL;
        layer.save_checkpoint(checkpoint,checkpoint_position);
        run("decode0", 1); run("decode1", 1);
        layer.restore_checkpoint(checkpoint);
        cuda_check(cudaDeviceSynchronize(), "synchronize E3B transaction restore");
        const auto restored_recurrent = download_bytes(layer.trace().state_after, layer.recurrent_state_bytes(), "download E3B restored recurrent state");
        require(restored_recurrent == snapshot_recurrent, "E3B transaction did not restore recurrent state exactly");
        require(download_bytes(layer.trace().state_before, layer.recurrent_state_bytes(), "download E3B restored recurrent trace") == snapshot_recurrent, "E3B transaction recurrent trace is stale after restore");
        layer.save_checkpoint(checkpoint,checkpoint_position); cuda_check(cudaDeviceSynchronize(), "synchronize E3B restored checkpoint");
        const auto restored_conv = download_bytes(checkpoint_conv.ptr, layer.physical_conv_state_bytes(), "download E3B restored physical convolution state");
        require(restored_conv == snapshot_conv, "E3B transaction did not restore all four convolution slots exactly");
        std::vector<std::byte> expected_conv_trace(layer.conv_state_bytes());
        for (std::size_t channel = 0; channel < QKV; ++channel) {
            std::copy_n(snapshot_conv.begin() + channel * 4 * sizeof(std::uint16_t),
                        3 * sizeof(std::uint16_t),
                        expected_conv_trace.begin() + channel * 3 * sizeof(std::uint16_t));
        }
        require(download_bytes(layer.conv_state_device(), layer.conv_state_bytes(), "download E3B restored convolution trace") == expected_conv_trace, "E3B transaction convolution trace is stale after restore");
        for (int step=0; step<DECODE_STEPS; ++step) {
            run("decode"+std::to_string(step), 1);
            require(download_bytes(output.ptr, HIDDEN * sizeof(std::uint16_t), "download E3B replay output") == reference_outputs[step], "E3B transaction output mismatch at decode" + std::to_string(step));
            layer.save_checkpoint(checkpoint,++checkpoint_position); cuda_check(cudaDeviceSynchronize(), "synchronize E3B replay checkpoint");
            require(download_bytes(checkpoint_recurrent.ptr, layer.recurrent_state_bytes(), "download E3B replay recurrent state") == reference_recurrent[step], "E3B transaction recurrent state mismatch at decode" + std::to_string(step));
            require(download_bytes(checkpoint_conv.ptr, layer.physical_conv_state_bytes(), "download E3B replay physical convolution state") == reference_conv[step], "E3B transaction physical convolution state mismatch at decode" + std::to_string(step));
        }
        std::cout << "TRANSACTION checkpoint_restore=PASS conv_slots=4 continuation_steps=" << DECODE_STEPS << " gpu_inference_only=PASS\n";
        // Re-run the complete prefill + decode sequence after reset and require bit-stable native replay.
        layer.reset(); run("prefill11", PREFILL);
        for (int step=0; step<DECODE_STEPS; ++step) run("decode"+std::to_string(step), 1);
        auto replay_state = download_state(layer); require(replay_state == native_states.back(), "E3B reset/replay recurrent state mismatch");
        std::cout << "RESET replay=PASS\n";
        run_gdn_retained_prefix_qualification(w, fixture, fixture_path);
        // Time a clean decode-like M=1 path. The input buffer still contains decode0, but
        // recurrent and convolution state are reset before the timing samples.
        layer.reset(); cuda_check(cudaDeviceSynchronize(), "synchronize E3B timing reset");
        layer.forward(static_cast<const std::uint16_t*>(input.ptr), static_cast<std::uint16_t*>(output.ptr), 1);
        cuda_check(cudaDeviceSynchronize(), "synchronize E3B timing warmup");
        std::vector<Exl3GdnLayerTimings> timings; for (int i=0;i<5;++i) {
            layer.forward(static_cast<const std::uint16_t*>(input.ptr), static_cast<std::uint16_t*>(output.ptr), 1, nullptr, true);
            cuda_check(cudaDeviceSynchronize(), "synchronize E3B timing sample");
            timings.push_back(layer.last_timings());
        }
        Exl3GdnLayerTimings avg; for (const auto& x:timings) { for (std::size_t i=0;i<avg.microseconds.size();++i) avg.microseconds[i]+=x.microseconds[i]; avg.total_microseconds+=x.total_microseconds; } for (double& x:avg.microseconds)x/=timings.size(); avg.total_microseconds/=timings.size();
        const char* labels[] = {"input norm","QKV projection","Z projection","control projections","convolution","recurrence/state update","GDN gated norm","output projection","residual + MLP norm","MLP gate + up","MLP activation","MLP down + final residual"};
        std::cout << "TIMING M=1 total_us=" << avg.total_microseconds << '\n'; for (std::size_t i=0;i<avg.microseconds.size();++i) std::cout << "TIMING M=1 " << labels[i] << " us=" << avg.microseconds[i] << '\n';
        require(prefill_metrics.back().rel <= .01 && worst_final_rel <= .01 && worst_state_rel <= .01, "E3B numerical gate failed");
        std::cout << "E3B native EXL3 projections=6 K={" << qkv.metadata.K << ',' << z.metadata.K << ',' << o.metadata.K << ',' << gate.metadata.K << ',' << up.metadata.K << ',' << down.metadata.K << "}\n";
        std::cout << "E3B workspace_bytes=" << layer.workspace_bytes() << "\nE3B GDN layer test: PASS\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "E3B GDN layer test: FAIL: " << e.what() << '\n'; return 1; }
}
