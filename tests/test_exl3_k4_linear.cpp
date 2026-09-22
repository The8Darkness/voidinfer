#include "exl3/linear_cuda.h"
#include "exl3/safetensors.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::exl3::Exl3CudaLinearMetadata;
using ninfer::exl3::Exl3CudaLinearWeights;
using ninfer::exl3::Exl3CudaLinearWorkspace;
using ninfer::exl3::TensorPayload;

constexpr int kInput = 5120;
constexpr int kOutput = 1024;
constexpr int kBits = 4;
constexpr float kHadamardScale = 0.08838834764831845f;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

std::string env_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string{};
}

TensorPayload load(const ninfer::exl3::IndexedSafetensors& collection,
                   const std::string& name) {
    for (const auto& shard : collection.shards)
        if (shard.header.find(name))
            return ninfer::exl3::read_tensor(shard.path, shard.header, name);
    throw std::runtime_error("missing tensor: " + name);
}

struct DeviceBuffer {
    void* data = nullptr;
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer() = default;
};

template<class T>
T* allocate(DeviceBuffer& buffer, std::size_t count, const char* label) {
    cuda_check(cudaMalloc(&buffer.data, count * sizeof(T)), label);
    return static_cast<T*>(buffer.data);
}

float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16u;
    std::uint32_t exponent = (h >> 10u) & 0x1fu;
    std::uint32_t fraction = h & 0x03ffu;
    std::uint32_t bits = sign;
    if (exponent == 0) {
        if (fraction) {
            unsigned shift = 0;
            while ((fraction & 0x0400u) == 0) { fraction <<= 1u; ++shift; }
            bits |= (127u - 14u - shift) << 23u;
            bits |= (fraction & 0x03ffu) << 13u;
        }
    } else if (exponent == 0x1fu) {
        bits |= 0x7f800000u | (fraction << 13u);
    } else {
        bits |= (exponent + 112u) << 23u | (fraction << 13u);
    }
    return std::bit_cast<float>(bits);
}

std::uint16_t float_to_half(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    const std::uint32_t absolute = bits & 0x7fffffffu;
    if (absolute >= 0x7f800000u)
        return static_cast<std::uint16_t>(sign | (absolute > 0x7f800000u ? 0x7e00u : 0x7c00u));
    int exponent = static_cast<int>((absolute >> 23u) & 0xffu) - 127 + 15;
    std::uint32_t mantissa = absolute & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<std::uint16_t>(sign);
        mantissa |= 0x800000u;
        const unsigned shift = static_cast<unsigned>(14 - exponent);
        std::uint32_t rounded = mantissa >> shift;
        const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    std::uint32_t rounded = mantissa >> 13u;
    const std::uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        if (++rounded == 0x400u) {
            rounded = 0;
            if (++exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<unsigned>(exponent) << 10u) | rounded);
}

std::uint16_t half_mul(std::uint16_t a, std::uint16_t b) {
    return float_to_half(half_to_float(a) * half_to_float(b));
}

std::uint16_t half_fma(std::uint16_t a, std::uint16_t b, std::uint16_t c) {
    return float_to_half(std::fma(half_to_float(a), half_to_float(b), half_to_float(c)));
}

void hadamard(float* values) {
    for (int width = 1; width < 128; width *= 2)
        for (int base = 0; base < 128; base += 2 * width)
            for (int i = 0; i < width; ++i) {
                const float left = values[base + i];
                const float right = values[base + width + i];
                values[base + i] = left + right;
                values[base + width + i] = left - right;
            }
}

constexpr std::array<std::uint16_t, 256> permutation() {
    std::array<std::uint16_t, 256> p{};
    for (int t = 0; t < 32; ++t) {
        const int r0 = (t % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
        const int c0 = t / 4, c1 = c0 + 8;
        p[t * 8 + 0] = static_cast<std::uint16_t>(r0 * 16 + c0);
        p[t * 8 + 1] = static_cast<std::uint16_t>(r1 * 16 + c0);
        p[t * 8 + 2] = static_cast<std::uint16_t>(r2 * 16 + c0);
        p[t * 8 + 3] = static_cast<std::uint16_t>(r3 * 16 + c0);
        p[t * 8 + 4] = static_cast<std::uint16_t>(r0 * 16 + c1);
        p[t * 8 + 5] = static_cast<std::uint16_t>(r1 * 16 + c1);
        p[t * 8 + 6] = static_cast<std::uint16_t>(r2 * 16 + c1);
        p[t * 8 + 7] = static_cast<std::uint16_t>(r3 * 16 + c1);
    }
    return p;
}

std::uint32_t load_u32(const std::uint16_t* words, int index) {
    return static_cast<std::uint32_t>(words[index * 2]) |
           (static_cast<std::uint32_t>(words[index * 2 + 1]) << 16u);
}

std::uint16_t decode_state(const std::uint16_t* packed, int t) {
    constexpr int words32 = kBits * 8;
    const int b0 = t * kBits + kBits - 16 + 256 * kBits;
    const int b1 = b0 + 16;
    const int shift = ((b1 - 1) / 32 + 1) * 32 - b1;
    const std::uint64_t merged =
        (static_cast<std::uint64_t>(load_u32(packed, (b0 / 32) % words32)) << 32u) |
        load_u32(packed, ((b1 - 1) / 32) % words32);
    return static_cast<std::uint16_t>((merged >> shift) & 0xffffu);
}

std::uint16_t decode_mul1(std::uint16_t state, std::uint32_t multiplier) {
    const std::uint32_t product = static_cast<std::uint32_t>(state) * multiplier;
    const std::uint32_t sum = (product & 0xffu) + ((product >> 8u) & 0xffu) +
                              ((product >> 16u) & 0xffu) + (product >> 24u);
    return half_fma(static_cast<std::uint16_t>(0x6400u + sum), 0x1eeeu, 0xc931u);
}

std::vector<std::uint16_t> reference(const std::vector<std::uint16_t>& input,
                                     const TensorPayload& trellis_payload,
                                     const TensorPayload& suh_payload,
                                     const TensorPayload& svh_payload,
                                     const TensorPayload& mul1_payload) {
    const auto trellis = trellis_payload.typed<std::uint16_t>("I16");
    const auto suh = suh_payload.typed<std::uint16_t>("F16");
    const auto svh = svh_payload.typed<std::uint16_t>("F16");
    const auto mul1 = mul1_payload.typed<std::int32_t>("I32");
    require(trellis_payload.info.shape == std::vector<std::uint64_t>{320, 64, 64},
            "unexpected K4 trellis shape");
    require(suh.size() == kInput && svh.size() == kOutput && mul1.size() == 1,
            "unexpected K4 auxiliary shape");

    std::vector<std::uint16_t> transformed(kInput);
    for (int block = 0; block < kInput; block += 128) {
        std::array<float, 128> values{};
        for (int i = 0; i < 128; ++i)
            values[i] = half_to_float(half_mul(input[block + i], suh[block + i]));
        hadamard(values.data());
        for (int i = 0; i < 128; ++i)
            transformed[block + i] = float_to_half(values[i] * kHadamardScale);
    }

    constexpr auto perm = permutation();
    std::vector<float> accum(kOutput, 0.0f);
    std::array<std::uint16_t, 256> tile{};
    for (int tile_n = 0; tile_n < kOutput / 16; ++tile_n) {
        for (int tile_k = 0; tile_k < kInput / 16; ++tile_k) {
            const auto* packed = trellis.data() +
                (static_cast<std::size_t>(tile_k) * (kOutput / 16) + tile_n) * 64u;
            for (int t = 0; t < 256; ++t)
                tile[perm[t]] = decode_mul1(decode_state(packed, t),
                    static_cast<std::uint32_t>(mul1[0]));
            for (int r = 0; r < 16; ++r) {
                const float x = half_to_float(transformed[tile_k * 16 + r]);
                for (int c = 0; c < 16; ++c)
                    accum[tile_n * 16 + c] += x * half_to_float(tile[r * 16 + c]);
            }
        }
    }

    std::vector<std::uint16_t> output(kOutput);
    for (int block = 0; block < kOutput; block += 128) {
        std::array<float, 128> values{};
        std::copy_n(accum.data() + block, 128, values.data());
        hadamard(values.data());
        for (int i = 0; i < 128; ++i)
            output[block + i] = half_mul(float_to_half(values[i] * kHadamardScale), svh[block + i]);
    }
    return output;
}

} // namespace

int main() {
    try {
        int devices = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status != cudaSuccess || devices == 0) {
            std::cerr << "SKIP: CUDA device unavailable\n";
            return 77;
        }
        const auto model = env_or_empty("NINFER_EXL3_TARGET_PATH");
        if (model.empty()) {
            std::cerr << "SKIP: set NINFER_EXL3_TARGET_PATH\n";
            return 77;
        }
        const auto collection = ninfer::exl3::inspect_indexed_directory(model);
        const std::string base = "mtp.layers.0.self_attn.k_proj.";
        auto trellis_payload = load(collection, base + "trellis");
        auto suh_payload = load(collection, base + "suh");
        auto svh_payload = load(collection, base + "svh");
        auto mul1_payload = load(collection, base + "mul1");

        std::vector<std::uint16_t> input(kInput);
        constexpr std::array<std::uint16_t, 8> values{
            0x3c00u, 0xbc00u, 0x3800u, 0xb800u, 0x3400u, 0xb400u, 0x3000u, 0xb000u};
        for (std::size_t i = 0; i < input.size(); ++i) input[i] = values[i % values.size()];
        const auto expected = reference(input, trellis_payload, suh_payload, svh_payload, mul1_payload);

        DeviceBuffer d_trellis, d_suh, d_svh, d_mul1, d_input, d_output;
        auto* trellis = allocate<std::uint16_t>(d_trellis, trellis_payload.bytes().size() / 2, "alloc trellis");
        auto* suh = allocate<std::uint16_t>(d_suh, kInput, "alloc suh");
        auto* svh = allocate<std::uint16_t>(d_svh, kOutput, "alloc svh");
        auto* mul1 = allocate<std::int32_t>(d_mul1, 1, "alloc mul1");
        auto* device_input = allocate<std::uint16_t>(d_input, kInput, "alloc input");
        auto* device_output = allocate<std::uint16_t>(d_output, kOutput + 2, "alloc output");
        cuda_check(cudaMemcpy(trellis, trellis_payload.bytes().data(), trellis_payload.bytes().size(), cudaMemcpyHostToDevice), "copy trellis");
        cuda_check(cudaMemcpy(suh, suh_payload.bytes().data(), suh_payload.bytes().size(), cudaMemcpyHostToDevice), "copy suh");
        cuda_check(cudaMemcpy(svh, svh_payload.bytes().data(), svh_payload.bytes().size(), cudaMemcpyHostToDevice), "copy svh");
        cuda_check(cudaMemcpy(mul1, mul1_payload.bytes().data(), 4, cudaMemcpyHostToDevice), "copy mul1");
        cuda_check(cudaMemcpy(device_input, input.data(), input.size() * 2, cudaMemcpyHostToDevice), "copy input");
        const std::vector<std::uint16_t> poison(kOutput + 2, 0x7bffu);
        cuda_check(cudaMemcpy(device_output, poison.data(), poison.size() * 2, cudaMemcpyHostToDevice), "poison output");

        Exl3CudaLinearWorkspace workspace(kInput, kOutput, 1);
        const Exl3CudaLinearWeights weights{trellis, suh, svh, mul1};
        const Exl3CudaLinearMetadata metadata{kInput, kOutput, kBits, false, true, false};
        require(std::string(workspace.dispatch_name(metadata, 1)) == "generic_tile",
                "K4 did not select the bounded generic tile route");
        workspace.forward(weights, metadata, device_input, device_output + 1, 1);
        workspace.forward(weights, metadata, device_input, device_output + 1, 1);
        cuda_check(cudaDeviceSynchronize(), "synchronize K4 result");

        std::vector<std::uint16_t> actual(kOutput + 2);
        std::vector<std::uint16_t> input_after(kInput);
        cuda_check(cudaMemcpy(actual.data(), device_output, actual.size() * 2, cudaMemcpyDeviceToHost), "copy output");
        cuda_check(cudaMemcpy(input_after.data(), device_input, input_after.size() * 2, cudaMemcpyDeviceToHost), "copy input after");
        require(actual.front() == 0x7bffu && actual.back() == 0x7bffu, "K4 output canary changed");
        require(input_after == input, "K4 input changed");
        double square_error = 0.0, square_reference = 0.0, maximum = 0.0;
        for (int i = 0; i < kOutput; ++i) {
            const double a = half_to_float(actual[i + 1]);
            const double e = half_to_float(expected[i]);
            const double d = std::abs(a - e);
            maximum = std::max(maximum, d);
            square_error += d * d;
            square_reference += e * e;
        }
        const double relative_l2 = std::sqrt(square_error / square_reference);
        require(maximum <= 0.002 && relative_l2 <= 0.0008,
                "K4 CUDA result differs from independent packed CPU reference");
        std::cout << "K4 dispatch=generic_tile max_abs=" << maximum
                  << " relative_l2=" << relative_l2 << "\n";

        constexpr int prefill_rows = 17;
        constexpr int timing_rows = 1024;
        std::vector<std::uint16_t> prefill_input(
            static_cast<std::size_t>(timing_rows) * kInput);
        for (std::size_t i = 0; i < prefill_input.size(); ++i)
            prefill_input[i] = values[i % values.size()];
        DeviceBuffer d_prefill_input, d_prefill_reference, d_prefill_candidate,
            d_prefill_mma, d_prefill_mma_repeat, d_prefill_native;
        auto* prefill_device_input = allocate<std::uint16_t>(
            d_prefill_input, prefill_input.size(), "alloc prefill input");
        auto* prefill_reference = allocate<std::uint16_t>(
            d_prefill_reference, static_cast<std::size_t>(prefill_rows) * kOutput,
            "alloc prefill reference");
        auto* prefill_candidate = allocate<std::uint16_t>(
            d_prefill_candidate, static_cast<std::size_t>(timing_rows) * kOutput,
            "alloc prefill candidate");
        const auto mma_guarded_count=static_cast<std::size_t>(timing_rows)*kOutput+2;
        auto* prefill_mma_guarded=allocate<std::uint16_t>(
            d_prefill_mma,mma_guarded_count,"alloc prefill MMA candidate");
        auto* prefill_mma_repeat_guarded=allocate<std::uint16_t>(
            d_prefill_mma_repeat,mma_guarded_count,"alloc prefill MMA repeat");
        auto* prefill_native_guarded=allocate<std::uint16_t>(
            d_prefill_native,mma_guarded_count,"alloc native MTP K4 route");
        auto* prefill_mma=prefill_mma_guarded+1;
        auto* prefill_mma_repeat=prefill_mma_repeat_guarded+1;
        auto* prefill_native=prefill_native_guarded+1;
        cuda_check(cudaMemset(prefill_mma_guarded,0xff,mma_guarded_count*sizeof(std::uint16_t)),
                   "initialize K4 MMA guards");
        cuda_check(cudaMemset(prefill_mma_repeat_guarded,0xff,mma_guarded_count*sizeof(std::uint16_t)),
                   "initialize K4 MMA repeat guards");
        cuda_check(cudaMemset(prefill_native_guarded,0xff,mma_guarded_count*sizeof(std::uint16_t)),
                   "initialize native MTP K4 route guards");
        cuda_check(cudaMemcpy(prefill_device_input, prefill_input.data(),
                              prefill_input.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice), "copy prefill input");
        Exl3CudaLinearWorkspace prefill_workspace(kInput, kOutput, timing_rows);
        using Admission = ninfer::exl3::Exl3CudaLinearAdmission;
        constexpr auto native_mtp = Admission::native_mtp_wide_prefill;
        constexpr auto native_mtp_one = Admission::native_mtp_one_step;
        require(std::string(prefill_workspace.dispatch_name(
                    metadata, prefill_rows, Admission::ordinary)) == "unsupported",
                "ordinary wide K4 must remain unsupported");
        require(prefill_workspace.native_mtp_wide_prefill_candidate(
                    metadata, prefill_rows, native_mtp),
                "native MTP K4 admission was rejected for a supported geometry");
        require(std::string(prefill_workspace.dispatch_name(
                    metadata, prefill_rows, native_mtp)) ==
                    "native_mtp_wide_prefill_mma64",
                "native MTP K4 admission did not select the N64 route");
        require(prefill_workspace.native_mtp_one_step_candidate(
                    metadata, 1, native_mtp_one) &&
                    std::string(prefill_workspace.dispatch_name(
                        metadata, 1, native_mtp_one)) ==
                        "native_mtp_one_step_mma64",
                "native MTP one-step K4 admission did not select the N64 route");
        require(!prefill_workspace.native_mtp_one_step_candidate(
                    metadata, 2, native_mtp_one),
                "native MTP one-step admission accepted multiple rows");
        for (int malformed = 0; malformed < 7; ++malformed) {
            auto bad = metadata;
            if (malformed == 0) bad.K = 5;
            if (malformed == 1) bad.mcg = true;
            if (malformed == 2) bad.mul1 = false;
            if (malformed == 3) bad.has_bias = true;
            if (malformed == 4) bad.in_features += 16;
            if (malformed == 5) bad.out_features += 128;
            if (malformed == 6) require(!prefill_workspace
                    .native_mtp_wide_prefill_candidate(metadata, 16, native_mtp),
                    "native MTP K4 admission ignored the wide-row bound");
            if (malformed != 6) {
                require(!prefill_workspace.native_mtp_wide_prefill_candidate(
                            bad, prefill_rows, native_mtp),
                        "malformed native MTP K4 geometry was admitted");
                require(std::string(prefill_workspace.dispatch_name(
                            bad, prefill_rows, native_mtp)) == "unsupported",
                        "malformed native MTP K4 geometry selected a route");
            }
        }
        prefill_workspace.forward(weights, metadata, prefill_device_input,
                                  prefill_reference, 16);
        prefill_workspace.forward(
            weights, metadata, prefill_device_input + 16 * kInput,
            prefill_reference + 16 * kOutput, 1);
        prefill_workspace.forward(weights, metadata, prefill_device_input,
                                  prefill_native, prefill_rows, nullptr,
                                  native_mtp);
        prefill_workspace.forward_k4_prefill_for_test(
            weights, metadata, prefill_device_input, prefill_candidate, prefill_rows);
        prefill_workspace.forward_k4_prefill_mma_for_test(
            weights, metadata, prefill_device_input, prefill_mma, prefill_rows);
        prefill_workspace.forward_k4_prefill_mma_for_test(
            weights, metadata, prefill_device_input, prefill_mma_repeat, prefill_rows);
        cuda_check(cudaDeviceSynchronize(), "synchronize K4 prefill differential");
        std::vector<std::uint16_t> prefill_reference_host(
            static_cast<std::size_t>(prefill_rows) * kOutput);
        std::vector<std::uint16_t> prefill_candidate_host(prefill_reference_host.size());
        std::vector<std::uint16_t> prefill_mma_host(prefill_reference_host.size());
        std::vector<std::uint16_t> prefill_mma_repeat_host(prefill_reference_host.size());
        std::vector<std::uint16_t> prefill_native_host(prefill_reference_host.size());
        cuda_check(cudaMemcpy(prefill_reference_host.data(), prefill_reference,
                              prefill_reference_host.size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost), "copy K4 prefill reference");
        cuda_check(cudaMemcpy(prefill_candidate_host.data(), prefill_candidate,
                              prefill_candidate_host.size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost), "copy K4 prefill candidate");
        cuda_check(cudaMemcpy(prefill_mma_host.data(),prefill_mma,
                              prefill_mma_host.size()*sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),"copy K4 MMA candidate");
        cuda_check(cudaMemcpy(prefill_mma_repeat_host.data(),prefill_mma_repeat,
                              prefill_mma_repeat_host.size()*sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),"copy K4 MMA repeat");
        cuda_check(cudaMemcpy(prefill_native_host.data(),prefill_native,
                              prefill_native_host.size()*sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),"copy native MTP K4 route");
        require(prefill_reference_host == prefill_candidate_host,
                "K4 wide-row discriminator differs from generic-tile baseline");
        require(prefill_mma_host==prefill_mma_repeat_host,
                "K4 MMA prefill discriminator is not deterministic");
        require(prefill_native_host == prefill_mma_host,
                "native MTP K4 route differs from qualified MMA discriminator");
        prefill_workspace.forward(weights, metadata, prefill_device_input,
                                  prefill_native, 1, nullptr, native_mtp_one);
        cuda_check(cudaDeviceSynchronize(),
                   "synchronize native MTP one-step K4 differential");
        std::vector<std::uint16_t> native_one_host(static_cast<std::size_t>(kOutput));
        cuda_check(cudaMemcpy(native_one_host.data(), prefill_native,
                              native_one_host.size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "copy native MTP one-step K4 output");
        double one_square_error=0.0,one_square_reference=0.0,one_maximum=0.0;
        for (int i=0;i<kOutput;++i) {
            const double expected_value=half_to_float(prefill_reference_host[i]);
            const double actual_value=half_to_float(native_one_host[i]);
            const double difference=std::abs(actual_value-expected_value);
            one_maximum=std::max(one_maximum,difference);
            one_square_error+=difference*difference;
            one_square_reference+=expected_value*expected_value;
        }
        const double one_relative_l2=std::sqrt(one_square_error/one_square_reference);
        require(one_maximum<=0.002 && one_relative_l2<=0.0008,
                "native MTP one-step K4 route failed declared tolerance");
        std::array<std::uint16_t,2> mma_guards{},mma_repeat_guards{},native_guards{};
        cuda_check(cudaMemcpy(mma_guards.data(),prefill_mma_guarded,sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),"copy K4 MMA leading guard");
        cuda_check(cudaMemcpy(mma_guards.data()+1,prefill_mma+timing_rows*kOutput,
                              sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
                   "copy K4 MMA trailing guard");
        cuda_check(cudaMemcpy(mma_repeat_guards.data(),prefill_mma_repeat_guarded,
                              sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
                   "copy K4 MMA repeat leading guard");
        cuda_check(cudaMemcpy(mma_repeat_guards.data()+1,
                              prefill_mma_repeat+timing_rows*kOutput,sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),"copy K4 MMA repeat trailing guard");
        cuda_check(cudaMemcpy(native_guards.data(),prefill_native_guarded,sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),"copy native MTP K4 leading guard");
        cuda_check(cudaMemcpy(native_guards.data()+1,
                              prefill_native+timing_rows*kOutput,sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),"copy native MTP K4 trailing guard");
        require(mma_guards==std::array<std::uint16_t,2>{0xffffu,0xffffu} &&
                mma_repeat_guards==std::array<std::uint16_t,2>{0xffffu,0xffffu} &&
                native_guards==std::array<std::uint16_t,2>{0xffffu,0xffffu},
                "K4 MMA prefill output guard changed");
        double mma_square_error=0.0,mma_square_reference=0.0,mma_maximum=0.0;
        std::size_t mma_mismatches=0;
        for(std::size_t i=0;i<prefill_reference_host.size();++i) {
            const double expected_value=half_to_float(prefill_reference_host[i]);
            const double actual_value=half_to_float(prefill_mma_host[i]);
            const double difference=std::abs(actual_value-expected_value);
            mma_maximum=std::max(mma_maximum,difference);
            mma_square_error+=difference*difference;
            mma_square_reference+=expected_value*expected_value;
            mma_mismatches+=prefill_reference_host[i]!=prefill_mma_host[i];
        }
        const double mma_relative_l2=std::sqrt(mma_square_error/mma_square_reference);
        std::cout << "K4 MMA rows=17 mismatches=" << mma_mismatches
                  << " max_abs=" << mma_maximum
                  << " relative_l2=" << mma_relative_l2 << '\n';
        require(mma_maximum<=0.0015 && mma_relative_l2<=0.0007,
                "K4 MMA prefill failed established EXL3 E1/E2 tolerance");

        for (int i = 0; i < 2; ++i)
            prefill_workspace.forward_k4_prefill_for_test(
                weights, metadata, prefill_device_input, prefill_candidate, timing_rows);
        cudaEvent_t start = nullptr, stop = nullptr;
        cuda_check(cudaEventCreate(&start), "create K4 prefill start");
        cuda_check(cudaEventCreate(&stop), "create K4 prefill stop");
        cuda_check(cudaEventRecord(start), "record K4 prefill start");
        constexpr int iterations = 5;
        for (int i = 0; i < iterations; ++i)
            prefill_workspace.forward_k4_prefill_for_test(
                weights, metadata, prefill_device_input, prefill_candidate, timing_rows);
        cuda_check(cudaEventRecord(stop), "record K4 prefill stop");
        cuda_check(cudaEventSynchronize(stop), "synchronize K4 prefill timing");
        float elapsed_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "read K4 prefill timing");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        for (int i = 0; i < 2; ++i)
            prefill_workspace.forward_k4_prefill_mma_for_test(
                weights, metadata, prefill_device_input, prefill_mma, timing_rows);
        cuda_check(cudaEventCreate(&start), "create K4 MMA prefill start");
        cuda_check(cudaEventCreate(&stop), "create K4 MMA prefill stop");
        cuda_check(cudaEventRecord(start), "record K4 MMA prefill start");
        for (int i = 0; i < iterations; ++i)
            prefill_workspace.forward_k4_prefill_mma_for_test(
                weights, metadata, prefill_device_input, prefill_mma, timing_rows);
        cuda_check(cudaEventRecord(stop), "record K4 MMA prefill stop");
        cuda_check(cudaEventSynchronize(stop), "synchronize K4 MMA prefill timing");
        float mma_elapsed_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&mma_elapsed_ms,start,stop),
                   "read K4 MMA prefill timing");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        std::cout << "K4 prefill rows=17 exact_bits=" << prefill_candidate_host.size()
                  << " rows=1024 median_like_us="
                  << (elapsed_ms * 1000.0f / iterations)
                  << " mma_us=" << (mma_elapsed_ms*1000.0f/iterations) << '\n';

        struct ProjectionSpec { const char* name; int input; int output; };
        constexpr std::array<ProjectionSpec,8> projections{{
            {"mtp.fc",10240,5120},
            {"mtp.layers.0.self_attn.q_proj",5120,12288},
            {"mtp.layers.0.self_attn.k_proj",5120,1024},
            {"mtp.layers.0.self_attn.v_proj",5120,1024},
            {"mtp.layers.0.self_attn.o_proj",6144,5120},
            {"mtp.layers.0.mlp.gate_proj",5120,17408},
            {"mtp.layers.0.mlp.up_proj",5120,17408},
            {"mtp.layers.0.mlp.down_proj",17408,5120},
        }};
        double projection_sum_us=0.0;
        bool all_projection_k4_tolerance=true;
        for(const auto& projection:projections) {
            const std::string projection_base=std::string(projection.name)+".";
            auto projection_trellis=load(collection,projection_base+"trellis");
            auto projection_suh=load(collection,projection_base+"suh");
            auto projection_svh=load(collection,projection_base+"svh");
            auto projection_mul1=load(collection,projection_base+"mul1");
            require(projection_trellis.info.shape==std::vector<std::uint64_t>{
                        static_cast<std::uint64_t>(projection.input/16),
                        static_cast<std::uint64_t>(projection.output/16),64},
                    std::string(projection.name)+" trellis shape");
            require(projection_suh.bytes().size()==static_cast<std::size_t>(projection.input)*2 &&
                    projection_svh.bytes().size()==static_cast<std::size_t>(projection.output)*2 &&
                    projection_mul1.bytes().size()==4,
                    std::string(projection.name)+" auxiliary shape");

            std::vector<std::uint16_t> shape_input(
                static_cast<std::size_t>(timing_rows)*projection.input);
            for(std::size_t i=0;i<shape_input.size();++i)
                shape_input[i]=values[i%values.size()];
            DeviceBuffer shape_trellis_buffer,shape_suh_buffer,shape_svh_buffer,
                shape_mul1_buffer,shape_input_buffer,shape_reference_buffer,
                shape_mma_buffer,shape_repeat_buffer;
            auto* shape_trellis=allocate<std::uint16_t>(shape_trellis_buffer,
                projection_trellis.bytes().size()/2,"alloc shape trellis");
            auto* shape_suh=allocate<std::uint16_t>(shape_suh_buffer,projection.input,
                "alloc shape suh");
            auto* shape_svh=allocate<std::uint16_t>(shape_svh_buffer,projection.output,
                "alloc shape svh");
            auto* shape_mul1=allocate<std::int32_t>(shape_mul1_buffer,1,"alloc shape mul1");
            auto* shape_input_device=allocate<std::uint16_t>(shape_input_buffer,
                shape_input.size(),"alloc shape input");
            auto* shape_reference=allocate<std::uint16_t>(shape_reference_buffer,
                static_cast<std::size_t>(timing_rows)*projection.output,
                "alloc shape reference");
            const auto shape_guarded_count=
                static_cast<std::size_t>(timing_rows)*projection.output+2;
            auto* shape_mma_guarded=allocate<std::uint16_t>(shape_mma_buffer,
                shape_guarded_count,"alloc shape MMA");
            auto* shape_repeat_guarded=allocate<std::uint16_t>(shape_repeat_buffer,
                static_cast<std::size_t>(prefill_rows)*projection.output+2,
                "alloc shape MMA repeat");
            auto* shape_mma=shape_mma_guarded+1;
            auto* shape_repeat=shape_repeat_guarded+1;
            cuda_check(cudaMemcpy(shape_trellis,projection_trellis.bytes().data(),
                                  projection_trellis.bytes().size(),cudaMemcpyHostToDevice),
                       "copy shape trellis");
            cuda_check(cudaMemcpy(shape_suh,projection_suh.bytes().data(),
                                  projection_suh.bytes().size(),cudaMemcpyHostToDevice),
                       "copy shape suh");
            cuda_check(cudaMemcpy(shape_svh,projection_svh.bytes().data(),
                                  projection_svh.bytes().size(),cudaMemcpyHostToDevice),
                       "copy shape svh");
            cuda_check(cudaMemcpy(shape_mul1,projection_mul1.bytes().data(),4,
                                  cudaMemcpyHostToDevice),"copy shape mul1");
            cuda_check(cudaMemcpy(shape_input_device,shape_input.data(),
                                  shape_input.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice),
                       "copy shape input");
            cuda_check(cudaMemset(shape_mma_guarded,0xff,
                                  shape_guarded_count*sizeof(std::uint16_t)),
                       "initialize shape MMA guards");
            cuda_check(cudaMemset(shape_repeat_guarded,0xff,
                                  (static_cast<std::size_t>(prefill_rows)*projection.output+2)*
                                      sizeof(std::uint16_t)),
                       "initialize shape repeat guards");

            const Exl3CudaLinearWeights shape_weights{
                shape_trellis,shape_suh,shape_svh,shape_mul1};
            const Exl3CudaLinearMetadata shape_metadata{
                projection.input,projection.output,4,false,true,false};
            Exl3CudaLinearWorkspace shape_workspace(
                projection.input,projection.output,timing_rows);
            shape_workspace.forward(shape_weights,shape_metadata,shape_input_device,
                                    shape_reference,16);
            shape_workspace.forward(shape_weights,shape_metadata,
                shape_input_device+16*projection.input,
                shape_reference+16*projection.output,1);
            shape_workspace.forward_k4_prefill_mma_for_test(
                shape_weights,shape_metadata,shape_input_device,shape_mma,prefill_rows);
            shape_workspace.forward_k4_prefill_mma_for_test(
                shape_weights,shape_metadata,shape_input_device,shape_repeat,prefill_rows);
            cuda_check(cudaDeviceSynchronize(),"synchronize all-shape K4 differential");

            const auto shape_values=static_cast<std::size_t>(prefill_rows)*projection.output;
            std::vector<std::uint16_t> shape_reference_host(shape_values),
                shape_mma_host(shape_values),shape_repeat_host(shape_values);
            cuda_check(cudaMemcpy(shape_reference_host.data(),shape_reference,
                                  shape_values*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
                       "copy shape reference");
            cuda_check(cudaMemcpy(shape_mma_host.data(),shape_mma,
                                  shape_values*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
                       "copy shape MMA");
            cuda_check(cudaMemcpy(shape_repeat_host.data(),shape_repeat,
                                  shape_values*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
                       "copy shape repeat");
            require(shape_mma_host==shape_repeat_host,
                    std::string(projection.name)+" MMA repeat mismatch");
            std::array<std::uint16_t,2> shape_guards{},shape_repeat_guards{};
            cuda_check(cudaMemcpy(shape_guards.data(),shape_mma_guarded,sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),"copy shape leading guard");
            cuda_check(cudaMemcpy(shape_guards.data()+1,
                                  shape_mma+timing_rows*projection.output,
                                  sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
                       "copy shape trailing guard");
            cuda_check(cudaMemcpy(shape_repeat_guards.data(),shape_repeat_guarded,
                                  sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
                       "copy shape repeat leading guard");
            cuda_check(cudaMemcpy(shape_repeat_guards.data()+1,
                                  shape_repeat+prefill_rows*projection.output,
                                  sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
                       "copy shape repeat trailing guard");
            require(shape_guards==std::array<std::uint16_t,2>{0xffffu,0xffffu} &&
                    shape_repeat_guards==std::array<std::uint16_t,2>{0xffffu,0xffffu},
                    std::string(projection.name)+" MMA guard changed");
            double shape_error=0.0,shape_reference_square=0.0,shape_maximum=0.0;
            std::size_t shape_mismatches=0;
            for(std::size_t i=0;i<shape_values;++i) {
                const double expected_value=half_to_float(shape_reference_host[i]);
                const double actual_value=half_to_float(shape_mma_host[i]);
                const double difference=std::abs(actual_value-expected_value);
                shape_maximum=std::max(shape_maximum,difference);
                shape_error+=difference*difference;
                shape_reference_square+=expected_value*expected_value;
                shape_mismatches+=shape_reference_host[i]!=shape_mma_host[i];
            }
            const double shape_relative_l2=shape_reference_square==0.0 ?
                (shape_error==0.0 ? 0.0 : INFINITY) :
                std::sqrt(shape_error/shape_reference_square);
            const bool shape_finite=std::isfinite(shape_maximum) &&
                std::isfinite(shape_relative_l2);
            const bool shape_e1e2=shape_finite && shape_maximum<=0.0015 &&
                shape_relative_l2<=0.0007;
            const bool shape_k4_tolerance=shape_finite && shape_maximum<=0.002 &&
                shape_relative_l2<=0.0008;
            all_projection_k4_tolerance=
                all_projection_k4_tolerance && shape_k4_tolerance;
            std::cout << "K4 MTP projection=" << projection.name
                      << " shape=" << projection.input << "x" << projection.output
                      << " mismatches=" << shape_mismatches
                      << " max_abs=" << shape_maximum
                      << " relative_l2=" << shape_relative_l2
                      << " legacy_e1e2=" << (shape_e1e2?"pass":"fail")
                      << " k4_tolerance=" << (shape_k4_tolerance?"pass":"fail") << '\n';

            shape_workspace.forward_k4_prefill_for_test(
                shape_weights,shape_metadata,shape_input_device,shape_reference,timing_rows);
            for(int i=0;i<2;++i)shape_workspace.forward_k4_prefill_mma_for_test(
                shape_weights,shape_metadata,shape_input_device,shape_mma,timing_rows);
            cudaEvent_t shape_start=nullptr,shape_stop=nullptr;
            cuda_check(cudaEventCreate(&shape_start),"create all-shape K4 start");
            cuda_check(cudaEventCreate(&shape_stop),"create all-shape K4 stop");
            cuda_check(cudaEventRecord(shape_start),"record all-shape K4 start");
            for(int i=0;i<iterations;++i)shape_workspace.forward_k4_prefill_mma_for_test(
                shape_weights,shape_metadata,shape_input_device,shape_mma,timing_rows);
            cuda_check(cudaEventRecord(shape_stop),"record all-shape K4 stop");
            cuda_check(cudaEventSynchronize(shape_stop),"synchronize all-shape K4 timing");
            float shape_elapsed_ms=0.0f;
            cuda_check(cudaEventElapsedTime(&shape_elapsed_ms,shape_start,shape_stop),
                       "read all-shape K4 timing");
            cudaEventDestroy(shape_start);cudaEventDestroy(shape_stop);
            const double shape_us=shape_elapsed_ms*1000.0/iterations;
            projection_sum_us+=shape_us;
            const auto shape_timing_values=
                static_cast<std::size_t>(timing_rows)*projection.output;
            std::vector<std::uint16_t> shape_timing_reference(shape_timing_values),
                shape_timing_mma(shape_timing_values);
            cuda_check(cudaMemcpy(shape_timing_reference.data(),shape_reference,
                                  shape_timing_values*sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),"copy M1024 shape reference");
            cuda_check(cudaMemcpy(shape_timing_mma.data(),shape_mma,
                                  shape_timing_values*sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),"copy M1024 shape MMA");
            double timing_error=0.0,timing_reference_square=0.0,timing_maximum=0.0;
            std::size_t timing_mismatches=0;
            for(std::size_t i=0;i<shape_timing_values;++i) {
                const double expected_value=half_to_float(shape_timing_reference[i]);
                const double actual_value=half_to_float(shape_timing_mma[i]);
                const double difference=std::abs(actual_value-expected_value);
                timing_maximum=std::max(timing_maximum,difference);
                timing_error+=difference*difference;
                timing_reference_square+=expected_value*expected_value;
                timing_mismatches+=shape_timing_reference[i]!=shape_timing_mma[i];
            }
            const double timing_relative_l2=timing_reference_square==0.0 ?
                (timing_error==0.0 ? 0.0 : INFINITY) :
                std::sqrt(timing_error/timing_reference_square);
            const bool timing_k4_tolerance=std::isfinite(timing_maximum) &&
                std::isfinite(timing_relative_l2) && timing_maximum<=0.002 &&
                timing_relative_l2<=0.0008;
            all_projection_k4_tolerance=
                all_projection_k4_tolerance && timing_k4_tolerance;
            std::vector<std::uint16_t> shape_input_after(shape_input.size());
            cuda_check(cudaMemcpy(shape_input_after.data(),shape_input_device,
                                  shape_input_after.size()*sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),"copy all-shape K4 input after");
            require(shape_input_after==shape_input,
                    std::string(projection.name)+" changed input");
            std::cout << "K4 MTP timing projection=" << projection.name
                      << " m1024_us=" << shape_us
                      << " mismatches=" << timing_mismatches
                      << " max_abs=" << timing_maximum
                      << " relative_l2=" << timing_relative_l2
                      << " k4_tolerance=" << (timing_k4_tolerance?"pass":"fail") << '\n';
        }
        std::cout << "K4 MTP all_projection_m1024_us=" << projection_sum_us
                  << " four_chunk_4096_projection_ms=" << projection_sum_us*4.0/1000.0
                  << '\n';
        require(all_projection_k4_tolerance,
                "one or more K4 MTP projections failed declared K4 tolerance");
        std::cout << "ninfer_exl3_k4_linear_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
