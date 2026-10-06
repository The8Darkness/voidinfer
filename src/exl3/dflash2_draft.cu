#include "exl3/gaming_operator_fixture.h"
#include "exl3/dflash2_draft.h"
#include "exl3/vericache_serving_coordinator.h"
#include "exl3/linear_workspace_requirements.h"
#include "exl3/text_model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <array>
#include <span>
#include <utility>

namespace ninfer::exl3 {
namespace {

using Json = nlohmann::json;

// ---- Draft geometry (validated against config.json / quantization_config.json). ----
constexpr int kHidden = 5120;
constexpr int kVocab = 248320;
constexpr int kLayers = 5;
constexpr int kQHeads = 32;
constexpr int kKVHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kIntermediate = 17408;
constexpr int kRopeDim = kHeadDim;
constexpr float kRopeTheta = 10000000.0f;
constexpr float kRmsEps = 1.0e-6f;
constexpr int kConvKernel = 2;
constexpr int kConvGroup = 16;
constexpr int kConvGroups = kHidden / kConvGroup;           // 320
constexpr int kConvDynamic = 2 * kConvKernel * kConvGroups; // 1280
constexpr int kRank = 256;
constexpr int kTopK = 16;
constexpr int kBlockCap = 8;
constexpr int kContextCap = 16; // E5A2 minimal draft window (GEMM engine rows <= 16)
constexpr int kKeyCap = kContextCap + kBlockCap;
constexpr int kRingCap = 2048;          // E5A3 ring slots (power of two)
constexpr int kRingMask = kRingCap - 1;
constexpr int kRingKeep = 2047;         // E5A3 max committed rows (sliding_window - 1)
constexpr std::array<int, kLayers> kTapLayers = {5, 19, 33, 47, 61};

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

__host__ __device__ float half_to_float(std::uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}
__host__ __device__ float bf16_to_float(std::uint16_t bits) {
    union { std::uint32_t u; float f; } value{static_cast<std::uint32_t>(bits) << 16u};
    return value.f;
}
__host__ __device__ std::uint16_t float_to_half(float value) {
    return __half_as_ushort(__float2half_rn(value));
}

// E5A2 numerical-range contract: the DFlash2 artifact declares dtype bfloat16
// and its residual stream genuinely operates at 1e3..1e5 scale (FP32 reference
// residM trajectory 38946/91210/108583/140893/52958). FP16 storage (max 65504)
// overflows at layer 1. The residual-adjacent buffers (x_a/x_b/convf/convf2)
// therefore use BF16 STORAGE with unchanged FP32 accumulation; every EXL3
// linear boundary stays F16 (workspace requirement, no kernel change).
enum class DFlashFmt : int { F16 = 0, BF16 = 1 };
__host__ __device__ std::uint16_t float_to_bf16(float value) {
    union { float f; std::uint32_t u; } v{value};
    if ((v.u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
    const std::uint32_t bias = 0x7fffu + ((v.u >> 16u) & 1u);
    v.u += bias;
    return static_cast<std::uint16_t>(v.u >> 16u);
}
__host__ __device__ float dflash_load(std::uint16_t bits, DFlashFmt fmt) {
    return fmt == DFlashFmt::BF16 ? bf16_to_float(bits) : half_to_float(bits);
}
__host__ __device__ std::uint16_t dflash_store(float value, DFlashFmt fmt) {
    return fmt == DFlashFmt::BF16 ? float_to_bf16(value) : float_to_half(value);
}

struct DeviceAllocation {
    struct Retained {
        std::optional<RetainedDeviceLedger::Ticket> device_credit;
        std::optional<RetainedDescriptorLedger::Ticket> metadata_credit;
        std::shared_ptr<const void> uncertain_source_owner;
        void* pointer=nullptr;std::size_t bytes=0;int device=-1,error=0;Retained* next=nullptr;
    };
    inline static std::atomic<Retained*> quarantine{nullptr};
    inline static std::atomic<std::uint64_t> quarantined_count{0};
    inline static std::atomic<std::uint64_t> native_cleanup_attempts{0};
    std::unique_ptr<Retained> retirement=std::make_unique<Retained>();
    std::optional<RetainedDescriptorLedger::Ticket> owner_metadata_credit;
    int device=-1;
    bool cleanup_failure_for_test=false;
    std::shared_ptr<const int> source_lifetime=std::make_shared<const int>(0);
    void* ptr = nullptr;
    std::size_t bytes = 0;
    static constexpr std::size_t metadata_bytes() noexcept {return sizeof(DeviceAllocation)+sizeof(Retained);}
    static bool attach_device_credit(const std::shared_ptr<const void>& owner,RetainedDeviceLedger::Ticket credit) noexcept {
        auto* value=const_cast<DeviceAllocation*>(static_cast<const DeviceAllocation*>(owner.get()));
        if(!value || !value->ptr || !value->retirement || value->retirement->device_credit || credit.bytes()!=value->bytes)return false;
        value->retirement->device_credit.emplace(std::move(credit));return true;
    }
    static bool attach_metadata_credit(const std::shared_ptr<const void>& owner,RetainedDescriptorLedger::Ticket credit) noexcept {
        auto* value=const_cast<DeviceAllocation*>(static_cast<const DeviceAllocation*>(owner.get()));
        if(!value || !value->retirement || value->retirement->metadata_credit || value->owner_metadata_credit ||
            credit.bytes()!=metadata_bytes())return false;
        auto record=credit.split(sizeof(Retained));if(!record)return false;
        value->retirement->metadata_credit.emplace(std::move(*record));
        value->owner_metadata_credit.emplace(std::move(credit));return true;
    }
    void prepare_device() {
        require(quarantined_count.load()==0,"unresolved draft allocation cleanup");
        cuda_check(cudaGetDevice(&device),"draft allocation device");
    }
    void release() noexcept {
        if(!ptr)return;
        if(!cleanup_failure_for_test && !uncertain_transfer_use)native_cleanup_attempts.fetch_add(1,std::memory_order_relaxed);
        int current=-1;auto error=(cleanup_failure_for_test || uncertain_transfer_use)?cudaErrorUnknown:cudaGetDevice(&current);
        if(error==cudaSuccess && current!=device)error=cudaErrorInvalidDevice;
        if(error==cudaSuccess)error=cudaFree(ptr);
        if(error!=cudaSuccess) {
            auto* record=retirement.release();
            record->pointer=ptr;record->bytes=bytes;record->device=device;record->error=static_cast<int>(error);
            if(uncertain_transfer_use)record->uncertain_source_owner=std::move(source_lifetime);
            auto* head=quarantine.load(std::memory_order_relaxed);
            do{record->next=head;}while(!quarantine.compare_exchange_weak(head,record,std::memory_order_release,std::memory_order_relaxed));
            quarantined_count.fetch_add(1,std::memory_order_release);
        }
        if(error==cudaSuccess && retirement)retirement->device_credit.reset();
        ptr=nullptr;
    }
    DeviceAllocation(std::span<const std::byte> source, const char* label) : bytes(source.size_bytes()) {
        prepare_device();
        cuda_check(cudaMalloc(&ptr, bytes), label);
        try {
            cuda_check(cudaMemcpy(ptr, source.data(), bytes, cudaMemcpyHostToDevice),
                       "upload DFlash2 tensor");
        } catch (...) {release();throw;}
    }
    explicit DeviceAllocation(std::size_t size, const char* label,
        std::optional<RetainedDeviceLedger::Ticket> device_credit={},
        std::optional<RetainedDescriptorLedger::Ticket> metadata_credit={}) : bytes(size) {
        require(bool(device_credit)==bool(metadata_credit),"draft constructor requires both credit domains");
        if(device_credit) {
            require(bytes && device_credit->bytes()==bytes && metadata_credit->bytes()==metadata_bytes(),
                "draft constructor credit extent mismatch");
            auto record=metadata_credit->split(sizeof(Retained));
            retirement->metadata_credit.emplace(std::move(*record));
            owner_metadata_credit.emplace(std::move(*metadata_credit));
            retirement->device_credit.emplace(std::move(*device_credit));
        }
        prepare_device();
        cuda_check(cudaMalloc(&ptr, bytes), label);
    }
    ~DeviceAllocation() {release();}
    void release_constructor_credits_after_commit() noexcept {
        retirement->device_credit.reset();retirement->metadata_credit.reset();owner_metadata_credit.reset();
    }
    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;
    bool uncertain_transfer_use=false;
};

std::span<const std::byte> as_bytes(std::span<const std::uint16_t> values) {
    return {reinterpret_cast<const std::byte*>(values.data()), values.size_bytes()};
}

TensorPayload load_tensor_from_file(const std::filesystem::path& path,
                                    const SafetensorsHeader& header,
                                    const std::string& name) {
    require(header.find(name) != nullptr, "DFlash2 tensor absent: " + name);
    return read_tensor(path, header, name);
}

const std::uint16_t* upload_f16_or_bf16(const TensorPayload& tensor, const char* label,
                                        std::vector<std::unique_ptr<DeviceAllocation>>& allocations,
                                        std::size_t& total) {
    require(tensor.info.dtype == "F16" || tensor.info.dtype == "BF16",
            tensor.info.name + " must be F16 or BF16");
    if (tensor.info.dtype == "F16") {
        void* p=nullptr;
        Exl3LinearWorkspaceRequirements::allocate_owned(allocations,total,&p,tensor.bytes().size_bytes(),
            [&]{return std::make_unique<DeviceAllocation>(tensor.bytes(),label);});
        return static_cast<const std::uint16_t*>(p);
    }
    const auto source = tensor.typed<std::uint16_t>("BF16");
    std::vector<std::uint16_t> converted;
    converted.reserve(source.size());
    for (const auto value : source) converted.push_back(float_to_half(bf16_to_float(value)));
    void* p=nullptr;
    Exl3LinearWorkspaceRequirements::allocate_owned(allocations,total,&p,as_bytes(converted).size_bytes(),
        [&]{return std::make_unique<DeviceAllocation>(as_bytes(converted),label);});
    return static_cast<const std::uint16_t*>(p);
}

const std::uint16_t* upload_raw16(const TensorPayload& tensor, const char* label,
                                  std::string_view dtype,
                                  std::vector<std::unique_ptr<DeviceAllocation>>& allocations,
                                  std::size_t& total) {
    require(tensor.info.dtype == dtype, tensor.info.name + " has unexpected dtype");
    void* p=nullptr;
    Exl3LinearWorkspaceRequirements::allocate_owned(allocations,total,&p,tensor.bytes().size_bytes(),
        [&]{return std::make_unique<DeviceAllocation>(tensor.bytes(),label);});
    return static_cast<const std::uint16_t*>(p);
}

const std::uint16_t* upload_kmajor_f16(
    const TensorPayload& tensor,int n,int k,const char* label,
    std::vector<std::unique_ptr<DeviceAllocation>>& allocations,std::size_t& total) {
    require(n>0 && k>0,"draft dense K-major geometry");
    const auto source=tensor.typed<std::uint16_t>("F16");
    require(source.size()==static_cast<std::size_t>(n)*k,
        tensor.info.name+" draft dense K-major extent");
    std::vector<std::uint16_t> transposed(source.size());
    for(int col=0;col<n;++col)for(int i=0;i<k;++i)
        transposed[static_cast<std::size_t>(i)*n+col]=
            source[static_cast<std::size_t>(col)*k+i];
    void* p=nullptr;
    Exl3LinearWorkspaceRequirements::allocate_owned(
        allocations,total,&p,as_bytes(transposed).size_bytes(),
        [&]{return std::make_unique<DeviceAllocation>(as_bytes(transposed),label);});
    return static_cast<const std::uint16_t*>(p);
}

// ---- E5A2 first-divergence host differential helpers (env-gated diagnostic). ----
// Host EXL3 K=5 decode that is bit-for-bit the same convention as the CUDA
// generic tile kernel and the qualified E2 host decode. It exists only to prove
// whether the device module is faithful for a non-canonical draft geometry
// (e.g. o_proj 4096->5120) on the identical real input, before any kernel edit.
struct HostLinearModule {
    std::vector<std::uint16_t> trellis; // [in/16][out/16][16*bits]
    std::vector<std::uint16_t> suh;     // [in]
    std::vector<std::uint16_t> svh;     // [out]
    int in = 0;
    int out = 0;
    std::uint32_t mul1 = 0;
};

std::uint16_t host_half_mul(std::uint16_t l, std::uint16_t r) {
    return float_to_half(half_to_float(l) * half_to_float(r));
}
std::uint16_t host_half_fma(std::uint16_t l, std::uint16_t r, std::uint16_t a) {
    return float_to_half(std::fma(half_to_float(l), half_to_float(r), half_to_float(a)));
}

void host_hadamard_128(float* values) {
    for (int width = 1; width < 128; width *= 2)
        for (int base = 0; base < 128; base += 2 * width)
            for (int i = 0; i < width; ++i) {
                const float left = values[base + i], right = values[base + width + i];
                values[base + i] = left + right;
                values[base + width + i] = left - right;
            }
}

constexpr std::array<std::uint16_t, 256> host_tensor_core_perm() {
    std::array<std::uint16_t, 256> permutation{};
    for (int t = 0; t < 32; ++t) {
        const int r0 = (t % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
        const int c0 = t / 4, c1 = c0 + 8;
        permutation[t * 8 + 0] = (std::uint16_t)(r0 * 16 + c0);
        permutation[t * 8 + 1] = (std::uint16_t)(r1 * 16 + c0);
        permutation[t * 8 + 2] = (std::uint16_t)(r2 * 16 + c0);
        permutation[t * 8 + 3] = (std::uint16_t)(r3 * 16 + c0);
        permutation[t * 8 + 4] = (std::uint16_t)(r0 * 16 + c1);
        permutation[t * 8 + 5] = (std::uint16_t)(r1 * 16 + c1);
        permutation[t * 8 + 6] = (std::uint16_t)(r2 * 16 + c1);
        permutation[t * 8 + 7] = (std::uint16_t)(r3 * 16 + c1);
    }
    return permutation;
}

std::uint32_t host_load_u32(const std::uint16_t* words, int index) {
    return (std::uint32_t)words[index * 2] | ((std::uint32_t)words[index * 2 + 1] << 16u);
}
std::uint16_t host_decode_state(const std::uint16_t* packed, int bits, int t_offset) {
    const int words32 = bits * 8;
    const int b0 = t_offset * bits + bits - 16 + 256 * bits;
    const int b1 = b0 + 16;
    const int shift = ((b1 - 1) / 32 + 1) * 32 - b1;
    const std::uint64_t merged =
        ((std::uint64_t)host_load_u32(packed, (b0 / 32) % words32) << 32u) |
        host_load_u32(packed, ((b1 - 1) / 32) % words32);
    return (std::uint16_t)((merged >> shift) & 0xffffu);
}
std::uint16_t host_decode_mul1(std::uint16_t state, std::uint32_t multiplier) {
    const std::uint32_t product = (std::uint32_t)state * multiplier;
    const std::uint32_t byte_sum = (product & 0xffu) + ((product >> 8u) & 0xffu) +
                                   ((product >> 16u) & 0xffu) + (product >> 24u);
    const auto input = (std::uint16_t)(0x6400u + byte_sum);
    return host_half_fma(input, (std::uint16_t)0x1eeeu, (std::uint16_t)0xc931u);
}

HostLinearModule read_host_linear(const std::filesystem::path& directory,
                                  const SafetensorsHeader& header,
                                  const std::string& prefix, int in, int out) {
    HostLinearModule w;
    w.in = in; w.out = out;
    const auto path = directory / "model.safetensors";
    const auto trellis = load_tensor_from_file(path, header, prefix + ".trellis");
    const auto suh = load_tensor_from_file(path, header, prefix + ".suh");
    const auto svh = load_tensor_from_file(path, header, prefix + ".svh");
    const auto mul1 = load_tensor_from_file(path, header, prefix + ".mul1");
    require(trellis.info.dtype == "I16" && suh.info.dtype == "F16" &&
                svh.info.dtype == "F16" && mul1.info.dtype == "I32",
            prefix + " differential tensor dtype mismatch");
    const auto tl = trellis.typed<std::uint16_t>("I16");
    w.trellis.assign(tl.begin(), tl.end());
    const auto sl = suh.typed<std::uint16_t>("F16");
    w.suh.assign(sl.begin(), sl.end());
    const auto vl = svh.typed<std::uint16_t>("F16");
    w.svh.assign(vl.begin(), vl.end());
    const auto ml = mul1.typed<std::int32_t>("I32");
    require(ml.size() == 1, prefix + " mul1 is not scalar");
    w.mul1 = static_cast<std::uint32_t>(ml.front());
    return w;
}

// rows x w.in (F16) -> rows x w.out (F16). Deterministic host EXL3 path.
std::vector<std::uint16_t> host_linear_forward(const HostLinearModule& w,
                                               const std::uint16_t* input,
                                               int rows) {
    constexpr int bits = 5;
    constexpr float kHScale = 0.088388347648f;
    const int tiles_k = w.in / 16, tiles_n = w.out / 16;
    const int packed_size = 16 * bits;
    constexpr auto permutation = host_tensor_core_perm();
    std::vector<std::uint16_t> transformed((size_t)rows * w.in);
    for (int row = 0; row < rows; ++row) {
        const std::uint16_t* xr = input + (size_t)row * w.in;
        for (int block = 0; block < w.in; block += 128) {
            std::array<float, 128> values{};
            for (int i = 0; i < 128; ++i)
                values[i] = half_to_float(host_half_mul(xr[block + i], w.suh[block + i]));
            host_hadamard_128(values.data());
            for (int i = 0; i < 128; ++i)
                transformed[(size_t)row * w.in + block + i] =
                    float_to_half(values[i] * kHScale);
        }
    }
    std::vector<float> accum((size_t)rows * w.out, 0.0f);
    std::array<std::uint16_t, 256> tile{};
    for (int tile_n = 0; tile_n < tiles_n; ++tile_n) {
        std::fill(tile.begin(), tile.end(), 0);
        for (int tile_k = 0; tile_k < tiles_k; ++tile_k) {
            const std::uint16_t* packed =
                w.trellis.data() + ((size_t)tile_k * tiles_n + tile_n) * packed_size;
            for (int t = 0; t < 256; ++t)
                tile[permutation[t]] = host_decode_mul1(host_decode_state(packed, bits, t), w.mul1);
            for (int row = 0; row < rows; ++row) {
                const std::uint16_t* x = transformed.data() + (size_t)row * w.in + tile_k * 16;
                float* y = accum.data() + (size_t)row * w.out + tile_n * 16;
                for (int r = 0; r < 16; ++r) {
                    const float xv = half_to_float(x[r]);
                    for (int c = 0; c < 16; ++c)
                        y[c] += xv * half_to_float(tile[r * 16 + c]);
                }
            }
        }
    }
    std::vector<std::uint16_t> output((size_t)rows * w.out);
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < w.out; block += 128) {
            std::array<float, 128> values{};
            std::copy_n(accum.data() + (size_t)row * w.out + block, 128, values.data());
            host_hadamard_128(values.data());
            for (int i = 0; i < 128; ++i) {
                const auto normalized = float_to_half(values[i] * kHScale);
                output[(size_t)row * w.out + block + i] =
                    host_half_mul(normalized, w.svh[block + i]);
            }
        }
    }
    return output;
}
// ---- E5A2 FP32 independent reference (decision gate; diagnostic only). ----
// Mathematically identical draft computation with FP32 activation storage and the
// SAME EXL3 linear oracle (host_linear_forward) used by the first-divergence
// differentials. Answers whether the observed thousands/tens-of-thousands
// trajectory is genuine model behavior (finite in FP32) or a shared
// decode-scale error (runaway even in FP32). Read-only w.r.t. the device path:
// inputs are downloaded, results only printed. No effect unless
// NINFER_DFLASH2_REF_FP32 is set (one-shot).
std::vector<float> ref32_from_f16(const std::vector<std::uint16_t>& h) {
    std::vector<float> f(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) f[i] = half_to_float(h[i]);
    return f;
}
std::vector<float> ref32_from_bf16(const std::vector<std::uint16_t>& h) {
    std::vector<float> f(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) f[i] = bf16_to_float(h[i]);
    return f;
}
std::vector<float> ref32_load_any(const std::filesystem::path& directory,
                                  const SafetensorsHeader& header,
                                  const std::string& name) {
    const auto path = directory / "model.safetensors";
    const auto tensor = load_tensor_from_file(path, header, name);
    std::vector<float> out;
    if (tensor.info.dtype == "F16") {
        const auto span = tensor.typed<std::uint16_t>("F16");
        out.reserve(span.size());
        for (const auto v : span) out.push_back(half_to_float(v));
    } else if (tensor.info.dtype == "BF16") {
        const auto span = tensor.typed<std::uint16_t>("BF16");
        out.reserve(span.size());
        for (const auto v : span) out.push_back(bf16_to_float(v));
    } else {
        require(false, "E5A2 ref32 unsupported weight dtype: " + name);
    }
    return out;
}
void ref32_rms(const std::vector<float>& in, const std::vector<float>& weight,
               std::vector<float>& out, int rows, int feats) {
    out.assign((std::size_t)rows * feats, 0.0f);
    for (int r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (int i = 0; i < feats; ++i) {
            const double v = in[(std::size_t)r * feats + i];
            sum += v * v;
        }
        const float inv = static_cast<float>(1.0 / std::sqrt(sum / feats + 1.0e-6));
        for (int i = 0; i < feats; ++i)
            out[(std::size_t)r * feats + i] = in[(std::size_t)r * feats + i] * inv * weight[i];
    }
}
void ref32_dense_t(const std::vector<float>& a, const std::vector<float>& w,
                   std::vector<float>& c, int rows, int k, int n) {
    c.assign((std::size_t)rows * n, 0.0f);
    for (int r = 0; r < rows; ++r)
        for (int col = 0; col < n; ++col) {
            double acc = 0.0;
            for (int i = 0; i < k; ++i)
                acc += static_cast<double>(a[(std::size_t)r * k + i]) * w[(std::size_t)col * k + i];
            c[(std::size_t)r * n + col] = static_cast<float>(acc);
        }
}
void ref32_dynconv(const std::vector<float>& x, const std::vector<float>& dyn,
                   const std::vector<float>& base, std::vector<float>& y,
                   int rows, int stream) {
    y.assign((std::size_t)rows * kHidden, 0.0f);
    const int per = kConvKernel * kConvGroups;
    for (int t = 0; t < rows; ++t)
        for (int i = 0; i < kHidden; ++i) {
            const int g = i / kConvGroup;
            const float x0 = x[(std::size_t)t * kHidden + i];
            const float x1 = t > 0 ? x[(std::size_t)(t - 1) * kHidden + i] : 0.0f;
            const float d0 = dyn[(std::size_t)t * kConvDynamic + stream * per + g];
            const float d1 = dyn[(std::size_t)t * kConvDynamic + stream * per + kConvGroups + g];
            const float b0 = base[(std::size_t)stream * (kConvKernel * kHidden) + i];
            const float b1 = base[(std::size_t)stream * (kConvKernel * kHidden) + kHidden + i];
            y[(std::size_t)t * kHidden + i] = (b0 + d0) * x0 + (b1 + d1) * x1;
        }
}
void ref32_headnorm(const std::vector<float>& in, const std::vector<float>& weight,
                    std::vector<float>& out, int segments) {
    out.assign(in.size(), 0.0f);
    for (int s = 0; s < segments; ++s) {
        double sum = 0.0;
        for (int i = 0; i < kHeadDim; ++i) {
            const double v = in[(std::size_t)s * kHeadDim + i];
            sum += v * v;
        }
        const float inv = static_cast<float>(1.0 / std::sqrt(sum / kHeadDim + 1.0e-6));
        for (int i = 0; i < kHeadDim; ++i)
            out[(std::size_t)s * kHeadDim + i] = in[(std::size_t)s * kHeadDim + i] * inv * weight[i];
    }
}
void ref32_rope(const std::vector<float>& in, std::vector<float>& out,
                std::span<const std::int32_t> pos, int rows, int heads) {
    out.assign(in.size(), 0.0f);
    const int half = kRopeDim / 2;
    for (int r = 0; r < rows; ++r)
        for (int h = 0; h < heads; ++h)
            for (int ch = 0; ch < kHeadDim; ++ch) {
                const std::size_t idx = ((std::size_t)r * heads + h) * kHeadDim + ch;
                if (ch >= kRopeDim) { out[idx] = in[idx]; continue; }
                const int pair = ch < half ? ch : ch - half;
                const int mate = ch < half ? ch + half : ch - half;
                const float angle = static_cast<float>(pos[r]) *
                    powf(kRopeTheta, -2.0f * static_cast<float>(pair) / static_cast<float>(kRopeDim));
                const float s = sinf(angle), c = cosf(angle);
                const float v = in[idx];
                const float vm = in[((std::size_t)r * heads + h) * kHeadDim + mate];
                out[idx] = v * c + (ch < half ? -vm : vm) * s;
            }
}
void ref32_attn(const std::vector<float>& q, const std::vector<float>& kctx,
                const std::vector<float>& vctx, const std::vector<float>& kblk,
                const std::vector<float>& vblk, std::vector<float>& out,
                int queries, int ctx_keys) {
    out.assign((std::size_t)queries * kQHeads * kHeadDim, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    std::vector<float> scores(64, 0.0f);
    for (int qq = 0; qq < queries; ++qq)
        for (int h = 0; h < kQHeads; ++h) {
            const int kvh = h / (kQHeads / kKVHeads);
            const int keys = ctx_keys + queries;
            float mx = -3.402823466e+38F;
            for (int k = 0; k < keys; ++k) {
                const float* kr = (k < ctx_keys)
                    ? &kctx[((std::size_t)k * kKVHeads + kvh) * kHeadDim]
                    : &kblk[((std::size_t)(k - ctx_keys) * kKVHeads + kvh) * kHeadDim];
                const float* qr = &q[((std::size_t)qq * kQHeads + h) * kHeadDim];
                float dot = 0.0f;
                for (int d = 0; d < kHeadDim; ++d) dot += qr[d] * kr[d];
                scores[k] = dot * scale;
                mx = std::max(mx, scores[k]);
            }
            float den = 0.0f;
            for (int k = 0; k < keys; ++k) { scores[k] = expf(scores[k] - mx); den += scores[k]; }
            for (int d = 0; d < kHeadDim; ++d) {
                float v = 0.0f;
                for (int k = 0; k < keys; ++k) {
                    const float* vr = (k < ctx_keys)
                        ? &vctx[((std::size_t)k * kKVHeads + kvh) * kHeadDim]
                        : &vblk[((std::size_t)(k - ctx_keys) * kKVHeads + kvh) * kHeadDim];
                    v += scores[k] / den * vr[d];
                }
                out[((std::size_t)qq * kQHeads + h) * kHeadDim + d] = v;
            }
        }
}
std::vector<float> ref32_linear(const HostLinearModule& w, const std::vector<float>& in, int rows) {
    std::vector<std::uint16_t> h(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) h[i] = float_to_half(in[i]);
    return ref32_from_f16(host_linear_forward(w, h.data(), rows));
}
void ref32_note(const std::string& label, const std::vector<float>& v, bool& ok) {
    double mx = 0.0, sum2 = 0.0;
    std::size_t nf = 0;
    for (float f : v) {
        if (!std::isfinite(f)) { ++nf; continue; }
        mx = std::max(mx, static_cast<double>(std::fabs(f)));
        sum2 += static_cast<double>(f) * f;
    }
    const double rms = v.empty() ? 0.0 : std::sqrt(sum2 / v.size());
    std::cout << "E5A2 ref_fp32 " << label << " maxabs=" << mx
              << " rms=" << rms << " nonfinite=" << nf << "/" << v.size() << "\n";
    if (nf > 0) ok = false;
}
void ref32_forward(const std::filesystem::path& directory, const SafetensorsHeader& header,
                   const std::vector<std::uint16_t>& xa_f16, const std::vector<std::uint16_t>& tcat_f16,
                   std::span<const std::int32_t> pos_ctx,std::span<const std::int32_t> pos_blk,
                   int block_len, int window_rows) {
    bool ok = true;
    std::vector<float> x = ref32_from_bf16(xa_f16);  // x_a is BF16 residual storage
    ref32_note("embed_xa", x, ok);
    const HostLinearModule fc = read_host_linear(directory, header, "fc", kLayers * kHidden, kHidden);
    const std::vector<float> fc_raw = ref32_from_f16(host_linear_forward(fc, tcat_f16.data(), window_rows));
    ref32_note("fc_raw", fc_raw, ok);
    const std::vector<float> hnorm_w = ref32_load_any(directory, header, "hidden_norm.weight");
    std::vector<float> hctx;
    ref32_rms(fc_raw, hnorm_w, hctx, window_rows, kHidden);
    ref32_note("hctx", hctx, ok);
    std::string traj;
    for (int li = 0; li < kLayers; ++li) {
        const std::string tag = "l" + std::to_string(li) + "_";
        const std::string base = "layers." + std::to_string(li);
        const HostLinearModule qm = read_host_linear(directory, header, base + ".self_attn.q_proj", kHidden, kQHeads * kHeadDim);
        const HostLinearModule km = read_host_linear(directory, header, base + ".self_attn.k_proj", kHidden, kKVHeads * kHeadDim);
        const HostLinearModule vm = read_host_linear(directory, header, base + ".self_attn.v_proj", kHidden, kKVHeads * kHeadDim);
        const HostLinearModule om = read_host_linear(directory, header, base + ".self_attn.o_proj", kQHeads * kHeadDim, kHidden);
        const HostLinearModule gm = read_host_linear(directory, header, base + ".mlp.gate_proj", kHidden, kIntermediate);
        const HostLinearModule um = read_host_linear(directory, header, base + ".mlp.up_proj", kHidden, kIntermediate);
        const HostLinearModule dm = read_host_linear(directory, header, base + ".mlp.down_proj", kIntermediate, kHidden);
        const std::vector<float> inorm = ref32_load_any(directory, header, base + ".input_layernorm.weight");
        const std::vector<float> postnorm = ref32_load_any(directory, header, base + ".post_attention_layernorm.weight");
        const std::vector<float> qnw = ref32_load_any(directory, header, base + ".self_attn.q_norm.weight");
        const std::vector<float> knw = ref32_load_any(directory, header, base + ".self_attn.k_norm.weight");
        const std::vector<float> attn_base = ref32_load_any(directory, header, base + ".attention_conv.base_kernel");
        const std::vector<float> attn_proj = ref32_load_any(directory, header, base + ".attention_conv.kernel_projection.weight");
        const std::vector<float> mlp_base = ref32_load_any(directory, header, base + ".mlp_conv.base_kernel");
        const std::vector<float> mlp_proj = ref32_load_any(directory, header, base + ".mlp_conv.kernel_projection.weight");
        const std::vector<float> kctx = ref32_linear(km, hctx, window_rows);
        const std::vector<float> vctx = ref32_linear(vm, hctx, window_rows);
        std::vector<float> kctx_n, kctx_r;
        ref32_headnorm(kctx, knw, kctx_n, window_rows * kKVHeads);
        ref32_rope(kctx_n, kctx_r, pos_ctx, window_rows, kKVHeads);
        std::vector<float> ln;
        ref32_rms(x, inorm, ln, block_len, kHidden);
        std::vector<float> dyn;
        ref32_dense_t(ln, attn_proj, dyn, block_len, kHidden, kConvDynamic);
        std::vector<float> conv;
        ref32_dynconv(ln, dyn, attn_base, conv, block_len, 0);
        const std::vector<float> qp = ref32_linear(qm, conv, block_len);
        const std::vector<float> kp = ref32_linear(km, conv, block_len);
        const std::vector<float> vp = ref32_linear(vm, conv, block_len);
        std::vector<float> qn, kn, qr, kr, attn, oproj;
        ref32_headnorm(qp, qnw, qn, block_len * kQHeads);
        ref32_headnorm(kp, knw, kn, block_len * kKVHeads);
        ref32_rope(qn, qr, pos_blk, block_len, kQHeads);
        ref32_rope(kn, kr, pos_blk, block_len, kKVHeads);
        ref32_attn(qr, kctx_r, vctx, kr, vp, attn, block_len, window_rows);
        oproj = ref32_linear(om, attn, block_len);
        ref32_note(tag + "oproj", oproj, ok);
        std::vector<float> convf;
        ref32_dynconv(oproj, dyn, attn_base, convf, block_len, 1);
        ref32_note(tag + "convf", convf, ok);
        std::vector<float> resA(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) resA[i] = x[i] + convf[i];
        ref32_note(tag + "residA", resA, ok);
        std::vector<float> ln2;
        ref32_rms(resA, postnorm, ln2, block_len, kHidden);
        std::vector<float> dyn2;
        ref32_dense_t(ln2, mlp_proj, dyn2, block_len, kHidden, kConvDynamic);
        std::vector<float> conv2;
        ref32_dynconv(ln2, dyn2, mlp_base, conv2, block_len, 0);
        const std::vector<float> gate = ref32_linear(gm, conv2, block_len);
        const std::vector<float> up = ref32_linear(um, conv2, block_len);
        std::vector<float> act(gate.size());
        for (std::size_t i = 0; i < gate.size(); ++i)
            act[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
        const std::vector<float> down = ref32_linear(dm, act, block_len);
        ref32_note(tag + "down", down, ok);
        std::vector<float> convf2;
        ref32_dynconv(down, dyn2, mlp_base, convf2, block_len, 1);
        std::vector<float> resM(resA.size());
        for (std::size_t i = 0; i < resA.size(); ++i) resM[i] = resA[i] + convf2[i];
        ref32_note(tag + "residM", resM, ok);
        double mx = 0.0;
        for (float f : resM) if (std::isfinite(f)) mx = std::max(mx, static_cast<double>(std::fabs(f)));
        if (!traj.empty()) traj += "/";
        traj += std::to_string(static_cast<long long>(mx));
        x = resM;
        std::cout.flush();
    }
    const std::vector<float> norm_w = ref32_load_any(directory, header, "norm.weight");
    std::vector<float> prefin(x.begin() + kHidden, x.end());
    ref32_note("pre_final", prefin, ok);
    std::vector<float> fn;
    ref32_rms(prefin, norm_w, fn, block_len - 1, kHidden);
    ref32_note("final_norm", fn, ok);
    const std::vector<float> hproj = ref32_load_any(directory, header, "candidate_selector.hidden_projection.weight");
    std::vector<float> hp;
    ref32_dense_t(fn, hproj, hp, block_len - 1, kHidden, kRank);
    ref32_note("hiddenproj_out", hp, ok);
    std::cout << "E5A2 ref_fp32 verdict=" << (ok ? "FINITE" : "NONFINITE")
              << " layers=" << kLayers << " residM_traj=" << traj << "\n";
    std::cout.flush();
}// ---- Native F16/BF16 draft kernels. ----

__global__ void dflash_embed_kernel(const std::int64_t* ids,
                                    const std::uint16_t* bf16_weight,
                                    std::uint16_t* output,
                                    int rows) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * kHidden;
    if (index >= total) return;
    const int row = index / kHidden;
    const int column = index % kHidden;
    const std::int64_t token = ids[row];
    // x_a is BF16 residual storage (artifact dtype contract).
    output[index] = float_to_bf16(bf16_to_float(bf16_weight[token * kHidden + column]));
}

// RMSNorm with RAW draft weights (the draft artifact stores unshifted norms; the
// target stores shifted norms and is NOT used here). Input storage is F16 except
// on the residual path (x_a/x_b use BF16); output is always F16 because every
// RMSNorm output is renormalized to O(1) scale. Accumulation stays FP32.
template <DFlashFmt kInFmt,bool Residual=false>
__global__ void __launch_bounds__(256) dflash_rms_norm_kernel(const std::uint16_t* input,
                                       const std::uint16_t* weight,
                                       std::uint16_t* output,
                                       int rows,
                                       int features,
                                       const std::uint16_t* right=nullptr,
                                       std::uint16_t* materialized=nullptr) {
    // 256 threads; each keeps its strided values in registers, and the sum
    // reproduces the 256-wide shared-memory tree (strides 128, 64, 32 through
    // shared memory, 16..1 by shuffles).
    constexpr int kThreads = 256;
    constexpr int kPerThread = 32;
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    extern __shared__ float shared[];
    const int lane = static_cast<int>(threadIdx.x);
    float values[kPerThread];
    float sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int i = lane + j * kThreads;
        if (i < features) {
            const int index=row*features+i;
            auto bits=input[index];
            if constexpr(Residual) {
                bits=float_to_bf16(bf16_to_float(bits)+bf16_to_float(right[index]));
                materialized[index]=bits;
            }
            values[j] = dflash_load(bits, kInFmt);
            sum += values[j] * values[j];
        }
    }
    shared[lane] = sum;
    __syncthreads();
    if (lane < 128) shared[lane] += shared[lane + 128];
    __syncthreads();
    if (lane < 64) shared[lane] += shared[lane + 64];
    __syncthreads();
    if (lane < 32) {
        float total = shared[lane] + shared[lane + 32];
        for (int offset = 16; offset > 0; offset >>= 1)
            total += __shfl_down_sync(0xffffffffu, total, offset);
        if (lane == 0) shared[0] = total;
    }
    __syncthreads();
    const float inv = rsqrtf(shared[0] / static_cast<float>(features) + kRmsEps);
    #pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int i = lane + j * kThreads;
        if (i < features)
            output[row * features + i] = float_to_half(values[j] * inv * half_to_float(weight[i]));
    }
}

// Grouped dynamic causal convolution. x/dyn/base are F16. Output storage is F16
// for the prepare stream and BF16 for the finish stream (convf/convf2 feed the
// BF16 residual path). Accumulation stays FP32.
// For each row t and channel i (group g = i/16):
//   y[t][i] = (base[s][0][i] + dyn[t][0][g]) * x[t][i]
//           + (base[s][1][i] + dyn[t][1][g]) * x[t-1][i]   (x[-1] == 0)
// dyn is the [rows][kConvDynamic] projection; s selects the stream half (0 prepare, 1 finish).
template <DFlashFmt kOutFmt,bool Residual=false>
__global__ void dflash_dyn_conv_kernel(const std::uint16_t* x,
                                       const std::uint16_t* dyn,
                                       const std::uint16_t* base, // [2][kConvKernel][kHidden]
                                       std::uint16_t* y,
                                       int rows,
                                       int stream,
                                       const std::uint16_t* residual=nullptr,
                                       std::uint16_t* residual_out=nullptr) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const std::uint16_t* base_row = base + stream * (kConvKernel * kHidden);
    // Grid (rows, kHidden / blockDim): one channel per thread.
    {
        const int i = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x);
        if (i >= kHidden) return;
        const int g = i / kConvGroup;
        const float x0 = half_to_float(x[row * kHidden + i]);
        const float x1 = row > 0 ? half_to_float(x[(row - 1) * kHidden + i]) : 0.0f;
        const float d0 = half_to_float(dyn[row * kConvDynamic +
                                          stream * (kConvKernel * kConvGroups) + g]);
        const float d1 = half_to_float(dyn[row * kConvDynamic +
                                          stream * (kConvKernel * kConvGroups) +
                                          kConvGroups + g]);
        const float b0 = half_to_float(base_row[i]);
        const float b1 = half_to_float(base_row[kHidden + i]);
        const auto bits=dflash_store((b0 + d0) * x0 + (b1 + d1) * x1, kOutFmt);
        y[row * kHidden + i] = bits;
        if constexpr(Residual)residual_out[row*kHidden+i]=float_to_bf16(
            bf16_to_float(residual[row*kHidden+i])+bf16_to_float(bits));
    }
}

// Residual add on the BF16 residual path (FP32 accumulation, BF16 storage).
__global__ void dflash_residual_kernel(const std::uint16_t* a,
                                       const std::uint16_t* b,
                                       std::uint16_t* out,
                                       int count) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    out[i] = float_to_bf16(bf16_to_float(a[i]) + bf16_to_float(b[i]));
}

__global__ void dflash_silu_mul_kernel(const std::uint16_t* gate,
                                       const std::uint16_t* up,
                                       std::uint16_t* out,
                                       int count) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float g = half_to_float(gate[i]);
    const float u = half_to_float(up[i]);
    out[i] = float_to_half((g / (1.0f + expf(-g))) * u);
}

// Dense F16 GEMM with the weight stored OUTPUT-major: B is [N][K], so
// C[m][n] = sum_i A[m][i] * B[n][i]. Used for the draft kernel_projection and the
// candidate-selector hidden_projection (both stored [out][in]).
template<bool WeightsKMajor>
__global__ void dflash_dense_gemm_t_kernel(const std::uint16_t* a,
                                           const std::uint16_t* b,
                                           std::uint16_t* c,
                                           int m, int k, int n) {
    const int row = static_cast<int>(blockIdx.y);
    // Every CTA owns one input row, but the old path rereads that row from
    // global memory independently for every output-column thread.  Stage the
    // represented F16 activation bits once; the loop below deliberately keeps
    // the original ascending FP32 accumulation order and weight addressing.
    extern __shared__ std::uint16_t shared_a[];
    for (int i = static_cast<int>(threadIdx.x); i < k; i += blockDim.x)
        shared_a[i] = a[static_cast<std::size_t>(row) * k + i];
    __syncthreads();
    for (int col = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
         col < n; col += static_cast<int>(gridDim.x) * blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < k; ++i) {
            const auto represented_weight=WeightsKMajor?
                b[static_cast<std::size_t>(i)*n+col]:
                b[static_cast<std::size_t>(col)*k+i];
            acc += half_to_float(shared_a[i]) * half_to_float(represented_weight);
        }
        c[row * n + col] = float_to_half(acc);
    }
}

// K-major dense rows kernel: 16 output columns x up to 8 rows per 128-thread
// CTA, with K tiles of both operands streamed through a cp.async ring. Every
// output keeps the single ascending FP32 chain of dflash_dense_gemm_t_kernel,
// so the stored F16 bits are identical; only operand delivery changes.
constexpr int kDenseRowsCols=8;
constexpr int kDenseRowsTileK=128;
constexpr int kDenseRowsStages=4;
constexpr int kDenseRowsMax=8;
constexpr int kDenseRowsThreads=kDenseRowsCols*kDenseRowsMax;
constexpr int kDenseRowsParts=kDenseRowsCols/8;

__device__ __forceinline__ void dflash_cp_async_16(void* shared_ptr,const void* global_ptr) {
    const auto shared_address=
        static_cast<std::uint32_t>(__cvta_generic_to_shared(shared_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(shared_address), "l"(global_ptr));
}

__global__ void __launch_bounds__(kDenseRowsThreads) dflash_dense_rows_kmajor_kernel(
    const std::uint16_t* a,const std::uint16_t* b,std::uint16_t* c,int m,int k,int n) {
    __shared__ __align__(16) std::uint16_t weights[kDenseRowsStages][kDenseRowsTileK][kDenseRowsCols];
    __shared__ __align__(16) std::uint16_t inputs[kDenseRowsStages][kDenseRowsMax][kDenseRowsTileK];
    const int tid=static_cast<int>(threadIdx.x);
    const int col0=static_cast<int>(blockIdx.x)*kDenseRowsCols;
    const int col=tid%kDenseRowsCols;
    const int row=tid/kDenseRowsCols;
    const int tiles=k/kDenseRowsTileK;
    auto load=[&](int tile,int stage) {
        const int k0=tile*kDenseRowsTileK;
        for(int chunk=tid;chunk<kDenseRowsTileK*kDenseRowsParts;chunk+=kDenseRowsThreads) {
            const int r=chunk/kDenseRowsParts,part=chunk%kDenseRowsParts;
            dflash_cp_async_16(&weights[stage][r][part*8],
                b+static_cast<std::size_t>(k0+r)*n+col0+part*8);
        }
        for(int chunk=tid;chunk<m*(kDenseRowsTileK/8);chunk+=kDenseRowsThreads) {
            const int r=chunk/(kDenseRowsTileK/8),part=chunk%(kDenseRowsTileK/8);
            dflash_cp_async_16(&inputs[stage][r][part*8],
                a+static_cast<std::size_t>(r)*k+k0+part*8);
        }
    };
    #pragma unroll
    for(int stage=0;stage<kDenseRowsStages-1;++stage) {
        if(stage<tiles) load(stage,stage);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    float acc=0.0f;
    for(int tile=0;tile<tiles;++tile) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(kDenseRowsStages-2));
        __syncthreads();
        const int next=tile+kDenseRowsStages-1;
        if(next<tiles) load(next,next%kDenseRowsStages);
        asm volatile("cp.async.commit_group;\n" ::);
        const int stage=tile%kDenseRowsStages;
        if(row<m) {
            #pragma unroll 4
            for(int i=0;i<kDenseRowsTileK;i+=8) {
                const uint4 packed=*reinterpret_cast<const uint4*>(&inputs[stage][row][i]);
                const auto* input_bits=reinterpret_cast<const std::uint16_t*>(&packed);
                #pragma unroll
                for(int j=0;j<8;++j)
                    acc+=half_to_float(input_bits[j])*half_to_float(weights[stage][i+j][col]);
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    if(row<m) c[static_cast<std::size_t>(row)*n+col0+col]=float_to_half(acc);
}

bool dflash_dense_rows_enabled() {
    static const bool enabled=[] {
        const char* value=std::getenv("NINFER_DFLASH2_DENSE_ROWS_PIPELINE");
        return !value || std::strcmp(value,"0")!=0;
    }();
    return enabled;
}

bool launch_dflash_dense_rows_kmajor(const std::uint16_t* in,const std::uint16_t* weights,
    std::uint16_t* out,int rows,int k,int n,cudaStream_t stream) {
    if(!dflash_dense_rows_enabled() || rows<1 || rows>kDenseRowsMax ||
       k%kDenseRowsTileK!=0 || n%kDenseRowsCols!=0 ||
       (reinterpret_cast<std::uintptr_t>(in)|reinterpret_cast<std::uintptr_t>(weights))%16!=0)
        return false;
    dflash_dense_rows_kmajor_kernel<<<n/kDenseRowsCols,kDenseRowsThreads,0,stream>>>(in,weights,out,rows,k,n);
    return true;
}

template<bool WeightsKMajor>
__global__ void gopt_draft_dense_rowpair_kernel(const std::uint16_t* a,
    const std::uint16_t* b,std::uint16_t* c,int m,int k,int n) {
    const int row=static_cast<int>(blockIdx.y)*2;
    const int col=static_cast<int>(blockIdx.x)*blockDim.x+threadIdx.x;
    extern __shared__ std::uint16_t shared_a[];
    for(int i=threadIdx.x;i<k;i+=blockDim.x) {
        shared_a[i]=a[static_cast<std::size_t>(row)*k+i];
        shared_a[k+i]=row+1<m?a[static_cast<std::size_t>(row+1)*k+i]:0;
    }
    __syncthreads();
    if(col>=n)return;
    float acc0=0.0f,acc1=0.0f;
    for(int i=0;i<k;++i) {
        const auto bits=WeightsKMajor?b[static_cast<std::size_t>(i)*n+col]:b[static_cast<std::size_t>(col)*k+i];
        const float weight=half_to_float(bits);
        acc0+=half_to_float(shared_a[i])*weight;
        acc1+=half_to_float(shared_a[k+i])*weight;
    }
    c[row*n+col]=float_to_half(acc0);
    if(row+1<m)c[(row+1)*n+col]=float_to_half(acc1);
}

// RoPE for the draft's own 128-dim rotary (half-split, theta 1e7). Per-row
// absolute positions come from device memory. input/output are [rows][heads][head_dim].
__global__ void dflash_rope_kernel(const std::uint16_t* input,
                                   std::uint16_t* output,
                                   const int* positions,
                                   int rows,
                                   int heads,
                                   int head_dim) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * heads * head_dim;
    if (index >= total) return;
    const int channel = index % head_dim;
    if (channel >= kRopeDim) { output[index] = input[index]; return; }
    const int half = kRopeDim / 2;
    const int pair = channel < half ? channel : channel - half;
    const int mate = channel < half ? channel + half : channel - half;
    const int logical_row = index / (heads * head_dim);
    const int head = (index / head_dim) % heads;
    const float angle = static_cast<float>(positions[logical_row]) *
        powf(kRopeTheta, -2.0f * static_cast<float>(pair) / static_cast<float>(kRopeDim));
    const float sine = sinf(angle);
    const float cosine = cosf(angle);
    const float v = half_to_float(input[index]);
    const float vm = half_to_float(input[logical_row * heads * head_dim + head * head_dim + mate]);
    const float r = v * cosine + (channel < half ? -vm : vm) * sine;
    output[index] = float_to_half(r);
}

__global__ void gopt_draft_rope_pair_kernel(const std::uint16_t* input,
    std::uint16_t* output,const int* positions,int rows,int heads) {
    const int i=static_cast<int>(blockIdx.x)*blockDim.x+threadIdx.x;
    constexpr int half=kRopeDim/2;
    if(i>=rows*heads*half)return;
    const int segment=i/half,pair=i%half;
    const int base=segment*kHeadDim;
    const float angle=static_cast<float>(positions[segment/heads])*
        powf(kRopeTheta,-2.0f*static_cast<float>(pair)/static_cast<float>(kRopeDim));
    const float sine=sinf(angle),cosine=cosf(angle);
    const float lo=half_to_float(input[base+pair]);
    const float hi=half_to_float(input[base+pair+half]);
    output[base+pair]=float_to_half(lo*cosine+(-hi)*sine);
    output[base+pair+half]=float_to_half(hi*cosine+lo*sine);
}

// Attention over S context keys + L block keys. The draft block is non-causal
// (config is_causal=false); the E5A2 context window never exceeds the sliding
// window, so no key masking is applied. One block per (query, head).
__global__ void dflash_attention_kernel(const std::uint16_t* q,     // [L][qheads][head_dim]
                                        const std::uint16_t* k_ctx, // [S][kv_heads][head_dim]
                                        const std::uint16_t* v_ctx,
                                        const std::uint16_t* k_blk, // [L][kv_heads][head_dim]
                                        const std::uint16_t* v_blk,
                                        std::uint16_t* out,         // [L][qheads][head_dim]
                                        int queries, int ctx_keys, int block_keys,
                                        float scale) {
    const int q_idx = static_cast<int>(blockIdx.x);
    const int query = q_idx / kQHeads;
    const int head = q_idx % kQHeads;
    if (query >= queries) return;
    const int kv_head = head / (kQHeads / kKVHeads);
    const int keys = ctx_keys + block_keys;
    float scores[64];
    float maximum = -3.402823466e+38F;
    const std::uint16_t* q_row = q + (query * kQHeads + head) * kHeadDim;
    for (int key = 0; key < keys; ++key) {
        const std::uint16_t* k_row = key < ctx_keys
            ? k_ctx + key * (kKVHeads * kHeadDim) + kv_head * kHeadDim
            : k_blk + (key - ctx_keys) * (kKVHeads * kHeadDim) + kv_head * kHeadDim;
        float dot = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) {
            dot += half_to_float(q_row[d]) * half_to_float(k_row[d]);
        }
        scores[key] = dot * scale;
        maximum = fmaxf(maximum, scores[key]);
    }
    float denominator = 0.0f;
    for (int key = 0; key < keys; ++key) {
        scores[key] = expf(scores[key] - maximum);
        denominator += scores[key];
    }
    for (int d = 0; d < kHeadDim; ++d) {
        float value = 0.0f;
        for (int key = 0; key < keys; ++key) {
            const std::uint16_t* v_row = key < ctx_keys
                ? v_ctx + key * (kKVHeads * kHeadDim) + kv_head * kHeadDim
                : v_blk + (key - ctx_keys) * (kKVHeads * kHeadDim) + kv_head * kHeadDim;
            value += scores[key] / denominator * half_to_float(v_row[d]);
        }
        out[(query * kQHeads + head) * kHeadDim + d] = float_to_half(value);
    }
}

// E5A3: scatter one K row (post-k_norm, post-RoPE) and one V row into ring slots.
// slot = absolute position mod 2048. One thread per element (1024).
__global__ void dflash_ring_scatter_kernel(const std::uint16_t* krow,
                                           const std::uint16_t* vrow,
                                           std::uint16_t* ring_k,
                                           std::uint16_t* ring_v,
                                           int slot) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int n = kKVHeads * kHeadDim;
    if (i >= n) return;
    ring_k[static_cast<std::size_t>(slot) * n + i] = krow[i];
    ring_v[static_cast<std::size_t>(slot) * n + i] = vrow[i];
}

__global__ void dflash_prefill_positions_kernel(std::int32_t* positions, int rows, int first) {
    const int row = static_cast<int>(threadIdx.x);
    if (row < rows) positions[row] = first + row;
}

__global__ void dflash_prefill_ring_scatter_kernel(const std::uint16_t* k,
    const std::uint16_t* v, std::uint16_t* ring_k, std::uint16_t* ring_v,
    int rows, int first) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int n = kKVHeads * kHeadDim;
    if (i >= rows * n) return;
    const int slot = (first + i / n) & kRingMask;
    const std::size_t dst = static_cast<std::size_t>(slot) * n + i % n;
    ring_k[dst] = k[i];
    ring_v[dst] = v[i];
}

// E5A3: attention over ring-committed ctx keys + block keys. Identical math to
// dflash_attention_kernel (FP32 accumulation, same scale, non-causal block) but
// with online softmax (no scores[64] stack limit) and modulo ring indexing, so it
// serves the full 2047-row committed window. All committed keys are within the
// 2048 sliding window of every block query by construction (ring holds <= 2047),
// exactly matching the reference mask (query - key < sliding_window).
__global__ void dflash_attention_ring_kernel(const std::uint16_t* q,
                                             const std::uint16_t* ring_k,
                                             const std::uint16_t* ring_v,
                                             int ring_start_slot,
                                             int ctx_keys,
                                             const std::uint16_t* k_blk,
                                             const std::uint16_t* v_blk,
                                             std::uint16_t* out,
                                             int queries, int block_keys,
                                             float scale) {
    const int q_idx = static_cast<int>(blockIdx.x);
    const int query = q_idx / kQHeads;
    const int head = q_idx % kQHeads;
    if (query >= queries) return;
    const int kv_head = head / (kQHeads / kKVHeads);
    const int kv_row = kKVHeads * kHeadDim;
    const std::uint16_t* q_row = q + (query * kQHeads + head) * kHeadDim;
    // Reference sliding mask (model_mlx.py DFlashAttention): ctx key i is visible
    // to block query j iff (ctx_keys + j - i) < sliding_window (2048). Older keys
    // are excluded; with count <= 2047 and block <= 8 this clips at most the 7
    // oldest keys for late block rows at a full ring. kRingCap == sliding_window.
    int ctx_start = ctx_keys + query - (kRingCap - 1);
    if (ctx_start < 0) ctx_start = 0;
    float maximum = -3.402823466e+38F;
    for (int key = ctx_start; key < ctx_keys; ++key) {
        const int slot = (ring_start_slot + key) & kRingMask;
        const std::uint16_t* k_row = ring_k + static_cast<std::size_t>(slot) * kv_row + kv_head * kHeadDim;
        float dot = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) dot += half_to_float(q_row[d]) * half_to_float(k_row[d]);
        maximum = fmaxf(maximum, dot * scale);
    }
    for (int key = 0; key < block_keys; ++key) {
        const std::uint16_t* k_row = k_blk + key * kv_row + kv_head * kHeadDim;
        float dot = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) dot += half_to_float(q_row[d]) * half_to_float(k_row[d]);
        maximum = fmaxf(maximum, dot * scale);
    }
    float denominator = 0.0f;
    float acc[128];
    for (int d = 0; d < kHeadDim; ++d) acc[d] = 0.0f;
    for (int key = ctx_start; key < ctx_keys; ++key) {
        const int slot = (ring_start_slot + key) & kRingMask;
        const std::uint16_t* k_row = ring_k + static_cast<std::size_t>(slot) * kv_row + kv_head * kHeadDim;
        const std::uint16_t* v_row = ring_v + static_cast<std::size_t>(slot) * kv_row + kv_head * kHeadDim;
        float dot = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) dot += half_to_float(q_row[d]) * half_to_float(k_row[d]);
        const float w = expf(dot * scale - maximum);
        denominator += w;
        for (int d = 0; d < kHeadDim; ++d) acc[d] += w * half_to_float(v_row[d]);
    }
    for (int key = 0; key < block_keys; ++key) {
        const std::uint16_t* k_row = k_blk + key * kv_row + kv_head * kHeadDim;
        const std::uint16_t* v_row = v_blk + key * kv_row + kv_head * kHeadDim;
        float dot = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) dot += half_to_float(q_row[d]) * half_to_float(k_row[d]);
        const float w = expf(dot * scale - maximum);
        denominator += w;
        for (int d = 0; d < kHeadDim; ++d) acc[d] += w * half_to_float(v_row[d]);
    }
    for (int d = 0; d < kHeadDim; ++d)
        out[(query * kQHeads + head) * kHeadDim + d] = float_to_half(acc[d] / denominator);
}

// Parallelize independent keys/columns, preserving the serial arithmetic order
// inside each dot, softmax denominator and output accumulator. One block/head.
__global__ void dflash_attention_ring_parallel_kernel(const std::uint16_t* q,
    const std::uint16_t* ring_k, const std::uint16_t* ring_v, int ring_start_slot,
    int ctx_keys, const std::uint16_t* k_blk, const std::uint16_t* v_blk,
    std::uint16_t* out, int queries, int block_keys, float scale) {
    const int q_idx = static_cast<int>(blockIdx.x);
    const int query = q_idx / kQHeads;
    const int head = q_idx % kQHeads;
    if (query >= queries) return;
    const int t = static_cast<int>(threadIdx.x);
    const int kv_head = head / (kQHeads / kKVHeads);
    const int kv_row = kKVHeads * kHeadDim;
    const auto* q_row = q + q_idx * kHeadDim;
    // Every key dot product in this block uses the same query row. Stage the
    // represented F16 values once so the key-parallel lanes do not reread the
    // same 128 global-memory elements for every assigned key. The dot-product
    // loop and its FP32 accumulation order are otherwise unchanged.
    __shared__ float query_values[kHeadDim];
    if (t < kHeadDim) query_values[t] = half_to_float(q_row[t]);
    __syncthreads();
    int ctx_start = ctx_keys + query - (kRingCap - 1);
    if (ctx_start < 0) ctx_start = 0;
    const int context = ctx_keys - ctx_start;
    const int keys = context + block_keys;
    __shared__ float scores[kRingKeep + kBlockCap];
    __shared__ float maximum;
    __shared__ float denominator;
    // Keep the context and block traversals separate.  The former is the hot
    // path for a full ring; splitting them removes the per-key source branch
    // and mixed pointer expression without changing the represented values or
    // the ascending key order stored in scores[].
    for (int key = t; key < context; key += kHeadDim) {
        const auto* k_row = ring_k + static_cast<std::size_t>(
            (ring_start_slot + ctx_start + key) & kRingMask) * kv_row +
            kv_head * kHeadDim;
        float dot = 0.0f;
        for (int d = 0; d < kHeadDim; ++d)
            dot += query_values[d] * half_to_float(k_row[d]);
        scores[key] = dot;
    }
    for (int key = context + t; key < keys; key += kHeadDim) {
        const auto* k_row = k_blk + (key - context) * kv_row + kv_head * kHeadDim;
        float dot = 0.0f;
        for (int d = 0; d < kHeadDim; ++d)
            dot += query_values[d] * half_to_float(k_row[d]);
        scores[key] = dot;
    }
    __syncthreads();
    if (t == 0) {
        float m = -3.402823466e+38F;
        for (int key = 0; key < keys; ++key) m = fmaxf(m, scores[key] * scale);
        maximum = m;
    }
    __syncthreads();
    for (int key = t; key < keys; key += kHeadDim)
        scores[key] = expf(scores[key] * scale - maximum);
    __syncthreads();
    if (t == 0) {
        float sum = 0.0f;
        for (int key = 0; key < keys; ++key) sum += scores[key];
        denominator = sum;
    }
    __syncthreads();
    float acc = 0.0f;
    // Match the original context-then-block accumulation order exactly.
    for (int key = 0; key < context; ++key) {
        const auto* v_row = ring_v + static_cast<std::size_t>(
            (ring_start_slot + ctx_start + key) & kRingMask) * kv_row +
            kv_head * kHeadDim;
        acc += scores[key] * half_to_float(v_row[t]);
    }
    for (int key = 0; key < block_keys; ++key) {
        const auto* v_row = v_blk + key * kv_row + kv_head * kHeadDim;
        acc += scores[context + key] * half_to_float(v_row[t]);
    }
    out[q_idx * kHeadDim + t] = float_to_half(acc / denominator);
}

// Staged ring attention: identical per-key dot, max, exp, denominator and
// chronological V accumulation as dflash_attention_ring_parallel_kernel, but K
// rows arrive through a coalesced shared tile, the max uses an exact
// order-free reduction, and V loads are issued eight keys ahead.
constexpr int kRingStageKeys=128;
constexpr int kRingStageDims=32;

__device__ __forceinline__ const std::uint16_t* dflash_ring_key_row(
    const std::uint16_t* ring, const std::uint16_t* blk, int ring_start_slot,
    int ctx_start, int context, int key, int kv_head) {
    constexpr int kv_row = kKVHeads * kHeadDim;
    return key < context ?
        ring + static_cast<std::size_t>((ring_start_slot + ctx_start + key) & kRingMask) *
            kv_row + kv_head * kHeadDim :
        blk + (key - context) * kv_row + kv_head * kHeadDim;
}

__global__ void __launch_bounds__(kHeadDim) dflash_attention_ring_staged_kernel(
    const std::uint16_t* q, const std::uint16_t* ring_k, const std::uint16_t* ring_v,
    int ring_start_slot, int ctx_keys, const std::uint16_t* k_blk,
    const std::uint16_t* v_blk, std::uint16_t* out, int queries, int block_keys,
    float scale) {
    const int q_idx = static_cast<int>(blockIdx.x);
    const int query = q_idx / kQHeads;
    const int head = q_idx % kQHeads;
    if (query >= queries) return;
    const int t = static_cast<int>(threadIdx.x);
    const int kv_head = head / (kQHeads / kKVHeads);
    const auto* q_row = q + q_idx * kHeadDim;
    __shared__ __align__(16) float query_values[kHeadDim];
    __shared__ __align__(16) float scores[kRingKeep + kBlockCap + 8];
    __shared__ __half2 key_tile[kRingStageKeys][kRingStageDims / 2 + 1];
    __shared__ float warp_max[kHeadDim / 32];
    __shared__ float maximum;
    __shared__ float denominator;
    query_values[t] = half_to_float(q_row[t]);
    int ctx_start = ctx_keys + query - (kRingCap - 1);
    if (ctx_start < 0) ctx_start = 0;
    const int context = ctx_keys - ctx_start;
    const int keys = context + block_keys;
    __syncthreads();
    constexpr int kParts = kRingStageDims / 8;
    for (int first = 0; first < keys; first += kRingStageKeys) {
        const int extent = min(kRingStageKeys, keys - first);
        float dot = 0.0f;
        for (int c = 0; c < kHeadDim; c += kRingStageDims) {
            for (int w = t; w < extent * kParts; w += kHeadDim) {
                const int local = w / kParts, part = w % kParts;
                const uint4 packed = *reinterpret_cast<const uint4*>(dflash_ring_key_row(
                    ring_k, k_blk, ring_start_slot, ctx_start, context, first + local,
                    kv_head) + c + part * 8);
                key_tile[local][part * 4 + 0] = *reinterpret_cast<const __half2*>(&packed.x);
                key_tile[local][part * 4 + 1] = *reinterpret_cast<const __half2*>(&packed.y);
                key_tile[local][part * 4 + 2] = *reinterpret_cast<const __half2*>(&packed.z);
                key_tile[local][part * 4 + 3] = *reinterpret_cast<const __half2*>(&packed.w);
            }
            __syncthreads();
            if (t < extent) {
                #pragma unroll
                for (int word = 0; word < kRingStageDims / 2; word += 2) {
                    const int d = c + 2 * word;
                    const float2 low = __half22float2(key_tile[t][word]);
                    const float2 high = __half22float2(key_tile[t][word + 1]);
                    const float4 q4 = *reinterpret_cast<const float4*>(&query_values[d]);
                    dot += q4.x * low.x;
                    dot += q4.y * low.y;
                    dot += q4.z * high.x;
                    dot += q4.w * high.y;
                }
            }
            __syncthreads();
        }
        if (t < extent) scores[first + t] = dot;
    }
    __syncthreads();
    // fmaxf is exact and order-free for non-NaN values, and any mix of signed
    // zeros yields identical exp(score*scale - max) values below.
    float local_max = -3.402823466e+38F;
    for (int key = t; key < keys; key += kHeadDim)
        local_max = fmaxf(local_max, scores[key] * scale);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, offset));
    if ((t & 31) == 0) warp_max[t >> 5] = local_max;
    __syncthreads();
    if (t == 0) {
        float m = warp_max[0];
        #pragma unroll
        for (int w = 1; w < kHeadDim / 32; ++w) m = fmaxf(m, warp_max[w]);
        maximum = m;
    }
    __syncthreads();
    for (int key = t; key < keys; key += kHeadDim)
        scores[key] = expf(scores[key] * scale - maximum);
    __syncthreads();
    if (t == 0) {
        float sum = 0.0f;
        for (int key = 0; key < keys; ++key) sum += scores[key];
        denominator = sum;
    }
    float acc = 0.0f;
    int key = 0;
    constexpr int kRingVBatch = 32;
    for (; key + kRingVBatch <= keys; key += kRingVBatch) {
        float values[kRingVBatch];
        #pragma unroll
        for (int j = 0; j < kRingVBatch; ++j)
            values[j] = half_to_float(dflash_ring_key_row(ring_v, v_blk, ring_start_slot,
                ctx_start, context, key + j, kv_head)[t]);
        #pragma unroll
        for (int j = 0; j < kRingVBatch; ++j) acc += scores[key + j] * values[j];
    }
    for (; key < keys; ++key)
        acc += scores[key] * half_to_float(dflash_ring_key_row(ring_v, v_blk,
            ring_start_slot, ctx_start, context, key, kv_head)[t]);
    __syncthreads();
    out[q_idx * kHeadDim + t] = float_to_half(acc / denominator);
}

bool dflash_ring_staged_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_DFLASH2_RING_STAGED");
        return !value || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

// Tensor-core split-K draft ring attention. One CTA per (64-key chunk, kv
// head) computes S = Q K^T for the kv head's 32 query rows (8 queries x 4 q
// heads) and the chunk's P V with m16n8k16 FP16 MMAs (FP32 accumulation),
// writing per-row max/sum and unnormalized FP32 outputs for the combine.
constexpr int kMmaChunk = 64;
constexpr int kMmaRows = kBlockCap * (kQHeads / kKVHeads);  // 32
constexpr int kMmaThreads = 256;
constexpr int kMmaStride = kHeadDim + 8;  // halves: rows 4 banks apart

__device__ __forceinline__ void dflash_mma16816(float (&c)[4], const std::uint32_t (&a)[4],
                                                std::uint32_t b0, std::uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(kMmaThreads) dflash_attention_ring_mma_kernel(
    const std::uint16_t* q, const std::uint16_t* ring_k, const std::uint16_t* ring_v,
    int ring_start_slot, int ctx_keys, const std::uint16_t* k_blk,
    const std::uint16_t* v_blk, int queries, int block_keys, float scale,
    float* partial_o, float* partial_ml) {
    constexpr int group = kQHeads / kKVHeads;
    const int chunk = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int chunks = static_cast<int>(gridDim.x);
    const int t = static_cast<int>(threadIdx.x);
    const int lane = t & 31, warp = t >> 5;
    const int g = lane >> 2, tig = lane & 3;
    const int rows = queries * group;
    const int lo = max(0, ctx_keys - (kRingCap - 1));
    const int context = ctx_keys - lo;
    const int keys = context + block_keys;
    const int first = chunk * kMmaChunk;
    __shared__ __align__(16) __half qs[kMmaRows][kMmaStride];
    __shared__ __align__(16) __half ks[kMmaChunk][kMmaStride];
    __shared__ __align__(16) __half vs[kMmaChunk][kMmaStride];
    // Scores reuse the K tile once every S fragment is in registers.
    static_assert(sizeof(float) * kMmaRows * (kMmaChunk + 4) <= sizeof(ks));
    auto ss = reinterpret_cast<float (*)[kMmaChunk + 4]>(&ks[0][0]);
    constexpr int kv_row = kKVHeads * kHeadDim;
    for (int i = t; i < kMmaRows * (kHeadDim / 8); i += kMmaThreads) {
        const int r = i / (kHeadDim / 8), part = i % (kHeadDim / 8);
        uint4 value = make_uint4(0, 0, 0, 0);
        if (r < rows) {
            const int q_idx = (r / group) * kQHeads + kv_head * group + r % group;
            value = *reinterpret_cast<const uint4*>(
                q + static_cast<std::size_t>(q_idx) * kHeadDim + part * 8);
        }
        *reinterpret_cast<uint4*>(&qs[r][part * 8]) = value;
    }
    for (int i = t; i < kMmaChunk * (kHeadDim / 8); i += kMmaThreads) {
        const int local = i / (kHeadDim / 8), part = i % (kHeadDim / 8);
        const int key = first + local;
        uint4 kp = make_uint4(0, 0, 0, 0), vp = make_uint4(0, 0, 0, 0);
        if (key < keys) {
            const std::size_t row = key < context
                ? static_cast<std::size_t>((ring_start_slot + lo + key) & kRingMask) * kv_row
                : static_cast<std::size_t>(key - context) * kv_row;
            const std::uint16_t* kb = key < context ? ring_k : k_blk;
            const std::uint16_t* vb = key < context ? ring_v : v_blk;
            kp = *reinterpret_cast<const uint4*>(kb + row + kv_head * kHeadDim + part * 8);
            vp = *reinterpret_cast<const uint4*>(vb + row + kv_head * kHeadDim + part * 8);
        }
        *reinterpret_cast<uint4*>(&ks[local][part * 8]) = kp;
        *reinterpret_cast<uint4*>(&vs[local][part * 8]) = vp;
    }
    __syncthreads();
    // S tiles: warp owns m-tile (warp & 1) and n-tiles 2 (warp >> 1) + {0, 1}.
    {
        const int m0 = (warp & 1) * 16;
        float c[2][4] = {};
        #pragma unroll
        for (int k0 = 0; k0 < kHeadDim; k0 += 16) {
            std::uint32_t a[4];
            a[0] = *reinterpret_cast<const std::uint32_t*>(&qs[m0 + g][k0 + 2 * tig]);
            a[1] = *reinterpret_cast<const std::uint32_t*>(&qs[m0 + g + 8][k0 + 2 * tig]);
            a[2] = *reinterpret_cast<const std::uint32_t*>(&qs[m0 + g][k0 + 8 + 2 * tig]);
            a[3] = *reinterpret_cast<const std::uint32_t*>(&qs[m0 + g + 8][k0 + 8 + 2 * tig]);
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                const int key = ((warp >> 1) * 2 + n) * 8 + g;
                dflash_mma16816(c[n], a,
                    *reinterpret_cast<const std::uint32_t*>(&ks[key][k0 + 2 * tig]),
                    *reinterpret_cast<const std::uint32_t*>(&ks[key][k0 + 8 + 2 * tig]));
            }
        }
        __syncthreads();
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int r = m0 + g + (e >> 1) * 8;
                const int local = ((warp >> 1) * 2 + n) * 8 + 2 * tig + (e & 1);
                const int key = first + local;
                // Query r / group sees context keys from ctx_keys + query - 2047 on.
                const bool valid = r < rows && key < keys &&
                    (key >= context || lo + key >= ctx_keys + r / group - (kRingCap - 1));
                ss[r][local] = valid ? c[n][e] * scale : -INFINITY;
            }
        }
    }
    __syncthreads();
    // Row max / exp / sum: warp w owns rows 4w..4w+3, two keys per lane.
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int r = warp * 4 + j;
        const float x0 = ss[r][lane], x1 = ss[r][lane + 32];
        float m = fmaxf(x0, x1);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        const float p0 = m == -INFINITY ? 0.0f : expf(x0 - m);
        const float p1 = m == -INFINITY ? 0.0f : expf(x1 - m);
        ss[r][lane] = p0; ss[r][lane + 32] = p1;
        float l = p0 + p1;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
        if (lane == 0 && r < rows) {
            float* ml = partial_ml +
                ((static_cast<std::size_t>(kv_head) * kMmaRows + r) * chunks + chunk) * 2;
            ml[0] = m; ml[1] = l;
        }
    }
    __syncthreads();
    // O tiles: warp owns m-tile (warp & 1) and d n-tiles 4 (warp >> 1) + {0..3}.
    {
        const int m0 = (warp & 1) * 16;
        float c[4][4] = {};
        #pragma unroll
        for (int k0 = 0; k0 < kMmaChunk; k0 += 16) {
            std::uint32_t a[4];
            const auto pack = [](float x, float y) {
                const __half2 h = __floats2half2_rn(x, y);
                return *reinterpret_cast<const std::uint32_t*>(&h);
            };
            a[0] = pack(ss[m0 + g][k0 + 2 * tig], ss[m0 + g][k0 + 2 * tig + 1]);
            a[1] = pack(ss[m0 + g + 8][k0 + 2 * tig], ss[m0 + g + 8][k0 + 2 * tig + 1]);
            a[2] = pack(ss[m0 + g][k0 + 8 + 2 * tig], ss[m0 + g][k0 + 9 + 2 * tig]);
            a[3] = pack(ss[m0 + g + 8][k0 + 8 + 2 * tig], ss[m0 + g + 8][k0 + 9 + 2 * tig]);
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int d = ((warp >> 1) * 4 + n) * 8 + g;
                const auto pair = [&](int key) {
                    const __half2 h = __halves2half2(vs[key][d], vs[key + 1][d]);
                    return *reinterpret_cast<const std::uint32_t*>(&h);
                };
                dflash_mma16816(c[n], a, pair(k0 + 2 * tig), pair(k0 + 8 + 2 * tig));
            }
        }
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int r = m0 + g + (e >> 1) * 8;
                const int d = ((warp >> 1) * 4 + n) * 8 + 2 * tig + (e & 1);
                if (r < rows)
                    partial_o[((static_cast<std::size_t>(kv_head) * kMmaRows + r) * chunks + chunk) *
                              kHeadDim + d] = c[n][e];
            }
        }
    }
}

// Combines the chunk partials of one (row, kv head) per CTA (FP32 rescale by
// exp(chunk max - row max), then one division).
__global__ void __launch_bounds__(kHeadDim) dflash_attention_ring_combine_kernel(
    const float* partial_o, const float* partial_ml, std::uint16_t* out, int queries,
    int chunks) {
    constexpr int group = kQHeads / kKVHeads;
    const int r = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    if (r >= queries * group) return;
    const int d = static_cast<int>(threadIdx.x);
    const std::size_t base = static_cast<std::size_t>(kv_head) * kMmaRows + r;
    const float* ml = partial_ml + base * chunks * 2;
    float m = -INFINITY;
    for (int c = 0; c < chunks; ++c) m = fmaxf(m, ml[2 * c]);
    float l = 0.0f, acc = 0.0f;
    const float* po = partial_o + base * chunks * kHeadDim;
    for (int c = 0; c < chunks; ++c) {
        const float mc = ml[2 * c];
        if (mc == -INFINITY) continue;
        const float w = expf(mc - m);
        l = fmaf(ml[2 * c + 1], w, l);
        acc = fmaf(po[static_cast<std::size_t>(c) * kHeadDim + d], w, acc);
    }
    const int q_idx = (r / group) * kQHeads + kv_head * group + r % group;
    out[static_cast<std::size_t>(q_idx) * kHeadDim + d] = float_to_half(acc / l);
}

// Split-K tensor-core route for long ring contexts (default; NINFER_DFLASH2_
// RING_SPLIT=0 keeps the per-(query, head) staged kernel). Short contexts stay
// on the staged kernel, which wins below kRingSplitMinKeys.
constexpr int kRingSplitMinKeys = 256;
constexpr int kRingSplitMaxChunks = (kRingKeep + kBlockCap + kMmaChunk - 1) / kMmaChunk;
constexpr std::size_t kRingSplitPartialFloats =
    static_cast<std::size_t>(kKVHeads) * kMmaRows * kRingSplitMaxChunks * kHeadDim;
constexpr std::size_t kRingSplitMlFloats =
    static_cast<std::size_t>(kKVHeads) * kMmaRows * kRingSplitMaxChunks * 2;

bool dflash_ring_split_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_DFLASH2_RING_SPLIT");
        return !value || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool dflash_ring_split_selected(int ctx_keys) {
    return dflash_ring_split_enabled() && ctx_keys >= kRingSplitMinKeys;
}

void launch_ring_attention_split(const std::uint16_t* q, const std::uint16_t* ring_k,
    const std::uint16_t* ring_v, int ring_start_slot, int ctx_keys,
    const std::uint16_t* k_blk, const std::uint16_t* v_blk, std::uint16_t* out,
    int queries, int block_keys, float scale, float* partial_o, float* partial_ml,
    cudaStream_t stream) {
    const int context = ctx_keys - std::max(0, ctx_keys - (kRingCap - 1));
    const int chunks = (context + block_keys + kMmaChunk - 1) / kMmaChunk;
    dflash_attention_ring_mma_kernel<<<dim3(chunks, kKVHeads), kMmaThreads, 0, stream>>>(
        q, ring_k, ring_v, ring_start_slot, ctx_keys, k_blk, v_blk, queries, block_keys,
        scale, partial_o, partial_ml);
    dflash_attention_ring_combine_kernel<<<dim3(queries * (kQHeads / kKVHeads), kKVHeads),
        kHeadDim, 0, stream>>>(partial_o, partial_ml, out, queries, chunks);
}

// E5A3: per-slot FNV-1a digest of ring K/V bytes, XOR-folded on host by ring_digest().
// One thread per slot; slot order is preserved by the host fold via slot labels.
__global__ void dflash_ring_digest_kernel(const std::uint16_t* ring_k,
                                          const std::uint16_t* ring_v,
                                          int start_slot, int slots,
                                          std::uint64_t* partial) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= slots) return;
    const int slot = (start_slot + i) & kRingMask;
    const int n = kKVHeads * kHeadDim;
    std::uint64_t h = 1469598103934665603ULL;
    h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(slot));
    h *= 1099511628211ULL;
    const std::uint16_t* k = ring_k + static_cast<std::size_t>(slot) * n;
    const std::uint16_t* v = ring_v + static_cast<std::size_t>(slot) * n;
    for (int e = 0; e < n; ++e) {
        h ^= static_cast<std::uint64_t>(k[e]);
        h *= 1099511628211ULL;
        h ^= static_cast<std::uint64_t>(v[e]) + 0x9e3779b97f4a7c15ULL;
        h *= 1099511628211ULL;
    }
    partial[i] = h;
}
} // namespace

// ===== E5A2 native DFlash2 draft kernels (correctness-first) =====

// Concatenate the five windowed target tap rows into [rows][5*5120] feature rows,
// ordering the tap layers exactly as the artifact config (5,19,33,47,61).
__global__ void dflash_tap_concat_kernel(const std::uint16_t* const* taps,
                                         std::uint16_t* dst, int rows) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * kLayers * kHidden;
    if (index >= total) return;
    const int segment = index / kHidden;
    const int row = segment / kLayers;
    const int tap = segment % kLayers;
    const int feature = index % kHidden;
    dst[row * (kLayers * kHidden) + tap * kHidden + feature] =
        taps[tap][row * kHidden + feature];
}

// RMSNorm over (row, head) segments of width head_dim (draft q/k norm, raw weights).
template<bool Rotate=false>
__global__ void dflash_head_norm_kernel(const std::uint16_t* input,
                                        const std::uint16_t* weight,
                                        std::uint16_t* output,
                                        int segments, int head_dim,
                                        const int* positions=nullptr,
                                        std::uint16_t* rotated=nullptr,int heads=0,
                                        bool paired=false) {
    __shared__ std::uint16_t represented[Rotate?128:1];
    const int segment = static_cast<int>(blockIdx.x);
    if (segment >= segments) return;
    extern __shared__ float shared[];
    const int lane = static_cast<int>(threadIdx.x);
    const std::uint16_t* base = input + static_cast<std::size_t>(segment) * head_dim;
    float sum = 0.0f;
    for (int i = lane; i < head_dim; i += blockDim.x) {
        const float value = half_to_float(base[i]);
        sum += value * value;
    }
    shared[lane] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) shared[lane] += shared[lane + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(shared[0] / static_cast<float>(head_dim) + kRmsEps);
    for (int i = lane; i < head_dim; i += blockDim.x) {
        const auto bits=float_to_half(half_to_float(base[i]) * inv * half_to_float(weight[i]));
        output[static_cast<std::size_t>(segment) * head_dim + i] = bits;
        if constexpr(Rotate)represented[i]=bits;
    }
    if constexpr(Rotate) {
        __syncthreads();
        constexpr int half=kRopeDim/2;
        if(lane<(paired?half:kRopeDim)) {
            const int pair=lane<half?lane:lane-half;
            const float angle=static_cast<float>(positions[segment/heads])*
                powf(kRopeTheta,-2.0f*static_cast<float>(pair)/static_cast<float>(kRopeDim));
            const float sine=sinf(angle),cosine=cosf(angle);
            const auto dst=static_cast<std::size_t>(segment)*head_dim;
            if(paired) {
                const float lo=half_to_float(represented[pair]);
                const float hi=half_to_float(represented[pair+half]);
                rotated[dst+pair]=float_to_half(lo*cosine+(-hi)*sine);
                rotated[dst+pair+half]=float_to_half(hi*cosine+lo*sine);
            } else {
                const float v=half_to_float(represented[lane]);
                const float vm=half_to_float(represented[lane<half?lane+half:lane-half]);
                rotated[dst+lane]=float_to_half(v*cosine+(lane<half?-vm:vm)*sine);
            }
        }
    }
}

// GOPT-012: the committed K/V ring is the only consumer of these represented
// values. Keep the baseline RMS reduction and RoPE expression, then write the
// resulting half directly to the ring beside its unmodified V element.
__global__ void gopt_draft_ring_norm_rope_kernel(const std::uint16_t* input,
    const std::uint16_t* vrow,const std::uint16_t* weight,const int* positions,
    std::uint16_t* ring_k,std::uint16_t* ring_v,int segments) {
    const int segment=static_cast<int>(blockIdx.x);
    if(segment>=segments)return;
    constexpr int heads=kKVHeads,dim=kHeadDim,half=kRopeDim/2;
    const int lane=static_cast<int>(threadIdx.x);
    extern __shared__ float shared[];
    __shared__ std::uint16_t represented[dim];
    const auto src=static_cast<std::size_t>(segment)*dim;
    const float value=half_to_float(input[src+lane]);
    shared[lane]=value*value;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1) {
        if(lane<stride)shared[lane]+=shared[lane+stride];
        __syncthreads();
    }
    const float inv=rsqrtf(shared[0]/static_cast<float>(dim)+kRmsEps);
    represented[lane]=float_to_half(value*inv*half_to_float(weight[lane]));
    __syncthreads();
    const int pair=lane<half?lane:lane-half;
    const int mate=lane<half?lane+half:lane-half;
    const int position=positions[segment/heads];
    const float angle=static_cast<float>(position)*
        powf(kRopeTheta,-2.0f*static_cast<float>(pair)/static_cast<float>(kRopeDim));
    const float sine=sinf(angle),cosine=cosf(angle);
    const float k=half_to_float(represented[lane]);
    const float km=half_to_float(represented[mate]);
    const float rotated=k*cosine+(lane<half?-km:km)*sine;
    const auto dst=static_cast<std::size_t>(position&kRingMask)*(heads*dim)+
        static_cast<std::size_t>(segment%heads)*dim;
    ring_k[dst+lane]=float_to_half(rotated);
    ring_v[dst+lane]=vrow[src+lane];
}

// In-place shift of a tap-history row buffer to drop the oldest rows.
__global__ void dflash_row_shift_kernel(const std::uint16_t* src, std::uint16_t* dst,
                                        int row_elems, int rows) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * row_elems;
    if (i < total) dst[i] = src[i];
}

// Preserve the original insertion sequence, including its NaN behavior.
__device__ void dflash_topk16_serial_row(const std::uint16_t* logits, int row, int vocab,
                                        std::int64_t* cand_ids, float* cand_unary) {
    float values[kTopK];
    int ids[kTopK];
    int count = 0;
    const std::uint16_t* row_logits = logits + static_cast<std::size_t>(row) * vocab;
    for (int column = 0; column < vocab; ++column) {
        const float value = half_to_float(row_logits[column]);
        if (count < kTopK) {
            int j = count;
            while (j > 0 && values[j - 1] < value) {
                values[j] = values[j - 1];
                ids[j] = ids[j - 1];
                --j;
            }
            values[j] = value;
            ids[j] = column;
            ++count;
        } else if (value > values[kTopK - 1]) {
            int j = kTopK - 1;
            while (j > 0 && values[j - 1] < value) {
                values[j] = values[j - 1];
                ids[j] = ids[j - 1];
                --j;
            }
            values[j] = value;
            ids[j] = column;
        }
    }
    for (int k = 0; k < kTopK; ++k) {
        cand_ids[static_cast<std::size_t>(row) * kTopK + k] = ids[k];
        cand_unary[static_cast<std::size_t>(row) * kTopK + k] = values[k];
    }
}

__global__ void dflash_topk16_kernel(const std::uint16_t* logits, int rows, int vocab,
                                    std::int64_t* cand_ids, float* cand_unary) {
    const int row = static_cast<int>(blockIdx.x);
    if (row < rows) dflash_topk16_serial_row(logits, row, vocab, cand_ids, cand_unary);
}

// Numeric ordering is total over (value descending, index ascending). Repeated
// argmax scans avoid a large per-thread top-K array and preserve input values.
// NaN rows retain the original serial GPU sequence instead of redefining it.
__global__ void dflash_topk16_parallel_kernel(const std::uint16_t* logits, int rows,
                                             int vocab, std::int64_t* cand_ids,
                                             float* cand_unary) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const int t = static_cast<int>(threadIdx.x);
    constexpr int invalid = 0x7fffffff;
    __shared__ float values[256];
    __shared__ int ids[256];
    __shared__ float previous_value;
    __shared__ int previous_id;
    const auto* input = logits + static_cast<std::size_t>(row) * vocab;
    if (t == 0) { previous_value = 0.0f; previous_id = -1; }
    __syncthreads();
    for (int rank = 0; rank < kTopK; ++rank) {
        float best = -CUDART_INF_F;
        int best_id = invalid;
        bool nan = false;
        for (int column = t; column < vocab; column += 256) {
            const float value = half_to_float(input[column]);
            nan = nan || isnan(value);
            const bool eligible = rank == 0 || value < previous_value ||
                (value == previous_value && column > previous_id);
            if (eligible && (value > best || (value == best && column < best_id))) {
                best = value;
                best_id = column;
            }
        }
        if (rank == 0 && __syncthreads_or(nan)) {
            if (t == 0) dflash_topk16_serial_row(logits, row, vocab, cand_ids, cand_unary);
            return;
        }
        values[t] = best;
        ids[t] = best_id;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (t < stride) {
                const float other = values[t + stride];
                const int other_id = ids[t + stride];
                if (other > values[t] || (other == values[t] && other_id < ids[t])) {
                    values[t] = other;
                    ids[t] = other_id;
                }
            }
            __syncthreads();
        }
        if (t == 0) {
            previous_value = values[0];
            previous_id = ids[0];
            cand_ids[static_cast<std::size_t>(row) * kTopK + rank] = ids[0];
            cand_unary[static_cast<std::size_t>(row) * kTopK + rank] = values[0];
        }
        __syncthreads();
    }
}

// One vocabulary read builds 256 private ordered runs. Lane zero of each warp
// merges its 32 runs, then thread zero merges the eight warp survivors. This is
// the same bounded merge shape used by the selector-lattice kernels, adapted to
// the draft head's FP16 logits and original int64 output contract.
template <bool TrackNonfinite>
__global__ void dflash_topk16_local_merge_kernel(const std::uint16_t* logits, int rows,
                                                 int vocab, std::int64_t* cand_ids,
                                                 float* cand_unary,
                                                 unsigned int* nonfinite_flag) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    constexpr int kThreads = 256;
    constexpr int kWarps = kThreads / 32;
    constexpr int kInvalid = 0x7fffffff;
    const int t = static_cast<int>(threadIdx.x);
    const auto* input = logits + static_cast<std::size_t>(row) * vocab;

    float local_values[kTopK];
    int local_ids[kTopK];
    #pragma unroll
    for (int i = 0; i < kTopK; ++i) {
        local_values[i] = -CUDART_INF_F;
        local_ids[i] = kInvalid;
    }
    bool nan = false;
    bool nonfinite = false;
    for (int column = t; column < vocab; column += kThreads) {
        const float value = half_to_float(input[column]);
        nan = nan || isnan(value);
        if constexpr (TrackNonfinite)
            nonfinite = nonfinite || value != value ||
                value == INFINITY || value == -INFINITY;
        if (value < local_values[kTopK - 1] ||
            (value == local_values[kTopK - 1] && column >= local_ids[kTopK - 1])) {
            continue;
        }
        int slot = kTopK - 1;
        while (slot > 0 &&
               (value > local_values[slot - 1] ||
                (value == local_values[slot - 1] && column < local_ids[slot - 1]))) {
            local_values[slot] = local_values[slot - 1];
            local_ids[slot] = local_ids[slot - 1];
            --slot;
        }
        local_values[slot] = value;
        local_ids[slot] = column;
    }
    if constexpr (TrackNonfinite) {
        if (__syncthreads_or(nonfinite) && t == 0)
            atomicExch(nonfinite_flag, 1u);
    }
    if (__syncthreads_or(nan)) {
        if (t == 0) dflash_topk16_serial_row(logits, row, vocab, cand_ids, cand_unary);
        return;
    }

    __shared__ float warp_values[kWarps * kTopK];
    __shared__ int warp_ids[kWarps * kTopK];
    for (int i = t; i < kWarps * kTopK; i += kThreads) {
        warp_values[i] = -CUDART_INF_F;
        warp_ids[i] = kInvalid;
    }
    __syncthreads();

    const int lane = t & 31;
    const int warp_base = (t >> 5) * kTopK;
    #pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        for (int source = 0; source < 32; ++source) {
            const float value = __shfl_sync(0xffffffffU, local_values[rank], source);
            const int id = __shfl_sync(0xffffffffU, local_ids[rank], source);
            if (lane != 0 || id == kInvalid ||
                value < warp_values[warp_base + kTopK - 1] ||
                (value == warp_values[warp_base + kTopK - 1] &&
                 id >= warp_ids[warp_base + kTopK - 1])) {
                continue;
            }
            int slot = kTopK - 1;
            while (slot > 0 &&
                   (value > warp_values[warp_base + slot - 1] ||
                    (value == warp_values[warp_base + slot - 1] &&
                     id < warp_ids[warp_base + slot - 1]))) {
                warp_values[warp_base + slot] = warp_values[warp_base + slot - 1];
                warp_ids[warp_base + slot] = warp_ids[warp_base + slot - 1];
                --slot;
            }
            warp_values[warp_base + slot] = value;
            warp_ids[warp_base + slot] = id;
        }
    }
    __syncthreads();

    if (t == 0) {
        float values[kTopK];
        int ids[kTopK];
        #pragma unroll
        for (int rank = 0; rank < kTopK; ++rank) {
            values[rank] = -CUDART_INF_F;
            ids[rank] = kInvalid;
        }
        for (int candidate = 0; candidate < kWarps * kTopK; ++candidate) {
            const float value = warp_values[candidate];
            const int id = warp_ids[candidate];
            if (id == kInvalid || value < values[kTopK - 1] ||
                (value == values[kTopK - 1] && id >= ids[kTopK - 1])) {
                continue;
            }
            int slot = kTopK - 1;
            while (slot > 0 &&
                   (value > values[slot - 1] ||
                    (value == values[slot - 1] && id < ids[slot - 1]))) {
                values[slot] = values[slot - 1];
                ids[slot] = ids[slot - 1];
                --slot;
            }
            values[slot] = value;
            ids[slot] = id;
        }
        #pragma unroll
        for (int rank = 0; rank < kTopK; ++rank) {
            cand_ids[static_cast<std::size_t>(row) * kTopK + rank] = ids[rank];
            cand_unary[static_cast<std::size_t>(row) * kTopK + rank] = values[rank];
        }
    }
}

// Segmented top-16: the (value desc, id asc) order is strict, so the top-16
// of the union of per-segment top-16 lists is exactly the row's top-16. Any
// NaN in a row routes the whole row to the serial kernel, as before.
constexpr int kTopKSegments = 32;
constexpr int kTopKSegmentThreads = 128;

__device__ __forceinline__ void dflash_topk16_insert(float* values, int* ids,
                                                     float value, int id) {
    constexpr int kInvalid = 0x7fffffff;
    if (id == kInvalid || value < values[kTopK - 1] ||
        (value == values[kTopK - 1] && id >= ids[kTopK - 1]))
        return;
    int slot = kTopK - 1;
    while (slot > 0 &&
           (value > values[slot - 1] || (value == values[slot - 1] && id < ids[slot - 1]))) {
        values[slot] = values[slot - 1];
        ids[slot] = ids[slot - 1];
        --slot;
    }
    values[slot] = value;
    ids[slot] = id;
}

// Lane zero merges the 32 register lists of its warp into shared[base..+16).
__device__ __forceinline__ void dflash_topk16_warp_merge(const float* local_values,
    const int* local_ids, float* shared_values, int* shared_ids) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    #pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        for (int source = 0; source < 32; ++source) {
            const float value = __shfl_sync(0xffffffffU, local_values[rank], source);
            const int id = __shfl_sync(0xffffffffU, local_ids[rank], source);
            if (lane == 0) dflash_topk16_insert(shared_values, shared_ids, value, id);
        }
    }
}

__global__ void __launch_bounds__(kTopKSegmentThreads) dflash_topk16_segment_kernel(
    const std::uint16_t* logits, int vocab, float* segment_values, int* segment_ids,
    int* segment_nan) {
    constexpr int kInvalid = 0x7fffffff;
    constexpr int kWarps = kTopKSegmentThreads / 32;
    const int segment = static_cast<int>(blockIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    const int t = static_cast<int>(threadIdx.x);
    const int span = (vocab + kTopKSegments - 1) / kTopKSegments;
    const int begin = segment * span;
    const int end = min(vocab, begin + span);
    const auto* input = logits + static_cast<std::size_t>(row) * vocab;
    float local_values[kTopK];
    int local_ids[kTopK];
    #pragma unroll
    for (int i = 0; i < kTopK; ++i) {
        local_values[i] = -CUDART_INF_F;
        local_ids[i] = kInvalid;
    }
    bool nan = false;
    for (int column = begin + t; column < end; column += kTopKSegmentThreads) {
        const float value = half_to_float(input[column]);
        nan = nan || isnan(value);
        dflash_topk16_insert(local_values, local_ids, value, column);
    }
    __shared__ float warp_values[kWarps * kTopK];
    __shared__ int warp_ids[kWarps * kTopK];
    for (int i = t; i < kWarps * kTopK; i += kTopKSegmentThreads) {
        warp_values[i] = -CUDART_INF_F;
        warp_ids[i] = kInvalid;
    }
    const bool any_nan = __syncthreads_or(nan);
    const int out = (row * kTopKSegments + segment) * kTopK;
    if (any_nan) {
        if (t == 0) segment_nan[row * kTopKSegments + segment] = 1;
        return;
    }
    const int base = (t >> 5) * kTopK;
    dflash_topk16_warp_merge(local_values, local_ids, warp_values + base, warp_ids + base);
    __syncthreads();
    if (t == 0) {
        float values[kTopK];
        int ids[kTopK];
        #pragma unroll
        for (int rank = 0; rank < kTopK; ++rank) {
            values[rank] = -CUDART_INF_F;
            ids[rank] = kInvalid;
        }
        for (int candidate = 0; candidate < kWarps * kTopK; ++candidate)
            dflash_topk16_insert(values, ids, warp_values[candidate], warp_ids[candidate]);
        #pragma unroll
        for (int rank = 0; rank < kTopK; ++rank) {
            segment_values[out + rank] = values[rank];
            segment_ids[out + rank] = ids[rank];
        }
        segment_nan[row * kTopKSegments + segment] = 0;
    }
}

__global__ void __launch_bounds__(32) dflash_topk16_segment_merge_kernel(
    const std::uint16_t* logits, int vocab, const float* segment_values,
    const int* segment_ids, const int* segment_nan, std::int64_t* cand_ids,
    float* cand_unary) {
    static_assert(kTopKSegments == 32, "one lane per segment");
    constexpr int kInvalid = 0x7fffffff;
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (__any_sync(0xffffffffU, segment_nan[row * kTopKSegments + lane] != 0)) {
        if (lane == 0) dflash_topk16_serial_row(logits, row, vocab, cand_ids, cand_unary);
        return;
    }
    float local_values[kTopK];
    int local_ids[kTopK];
    const int in = (row * kTopKSegments + lane) * kTopK;
    #pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        local_values[rank] = segment_values[in + rank];
        local_ids[rank] = segment_ids[in + rank];
    }
    __shared__ float values[kTopK];
    __shared__ int ids[kTopK];
    if (lane < kTopK) {
        values[lane] = -CUDART_INF_F;
        ids[lane] = kInvalid;
    }
    __syncwarp();
    dflash_topk16_warp_merge(local_values, local_ids, values, ids);
    __syncwarp();
    if (lane < kTopK) {
        cand_ids[static_cast<std::size_t>(row) * kTopK + lane] = ids[lane];
        cand_unary[static_cast<std::size_t>(row) * kTopK + lane] = values[lane];
    }
}


// Chunked exact top-16 (default; NINFER_DFLASH2_TOPK_CHUNKED=0 keeps the
// segmented insertion route). With T the 16th largest of the per-128-logit
// chunk maxima, every top-16 logit lies in a chunk whose maximum is >= T, so
// sorting the (value desc, id asc) pairs of those chunks yields exactly the
// serial insertion order. NaN rows and rows with more than kTopKChunkCandidates
// such chunks (ties) take dflash_topk16_serial_row.
constexpr int kTopKChunk = 128;
constexpr int kTopKMaxChunks = (kVocab + 256 + kTopKChunk - 1) / kTopKChunk;
constexpr int kTopKChunkSort = 2048;  // chunk-maximum capacity per row
constexpr int kTopKChunkCandidates = 32;
constexpr int kTopKCandidateSort = kTopKChunkCandidates * kTopKChunk;
constexpr int kTopKChunkThreads = 256;
static_assert(kTopKMaxChunks <= kTopKChunkSort);

__global__ void __launch_bounds__(kTopKChunkThreads) dflash_topk16_chunk_max_kernel(
    const std::uint16_t* logits, int vocab, float* chunk_max, int* chunk_nan) {
    constexpr int kWarps = kTopKChunkThreads / 32;
    const int row = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int chunk = static_cast<int>(blockIdx.x) * kWarps + (static_cast<int>(threadIdx.x) >> 5);
    const int chunks = (vocab + kTopKChunk - 1) / kTopKChunk;
    if (chunk >= chunks) return;
    const auto* input = logits + static_cast<std::size_t>(row) * vocab;
    float maximum = -CUDART_INF_F;
    bool nan = false;
    #pragma unroll
    for (int i = 0; i < kTopKChunk / 32; ++i) {
        const int column = chunk * kTopKChunk + lane + 32 * i;
        if (column < vocab) {
            const float value = half_to_float(input[column]);
            nan = nan || isnan(value);
            maximum = fmaxf(maximum, value);
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
    nan = __any_sync(0xffffffffu, nan);
    if (lane == 0) {
        chunk_max[static_cast<std::size_t>(row) * kTopKMaxChunks + chunk] = maximum;
        chunk_nan[static_cast<std::size_t>(row) * kTopKMaxChunks + chunk] = nan ? 1 : 0;
    }
}

// a precedes b in (value desc, id asc) order.
__device__ __forceinline__ bool dflash_topk_precedes(float av, int ai, float bv, int bi) {
    return av > bv || (av == bv && ai < bi);
}

// Block-wide first pair in (value desc, id asc) order; every thread gets it.
__device__ __forceinline__ void dflash_block_first(float& value, int& id, float* warp_values,
                                                   int* warp_ids) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float v = __shfl_xor_sync(0xffffffffu, value, offset);
        const int i = __shfl_xor_sync(0xffffffffu, id, offset);
        if (dflash_topk_precedes(v, i, value, id)) { value = v; id = i; }
    }
    if (lane == 0) { warp_values[warp] = value; warp_ids[warp] = id; }
    __syncthreads();
    value = warp_values[lane];
    id = warp_ids[lane];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float v = __shfl_xor_sync(0xffffffffu, value, offset);
        const int i = __shfl_xor_sync(0xffffffffu, id, offset);
        if (dflash_topk_precedes(v, i, value, id)) { value = v; id = i; }
    }
    __syncthreads();
}

constexpr int kTopKSelectThreads = 1024;
constexpr int kTopKSelectPerThread = kTopKCandidateSort / kTopKSelectThreads;
static_assert(kTopKSelectThreads == 32 * 32, "two-level warp reduction");
static_assert(kTopKChunkSort <= kTopKSelectThreads * kTopKSelectPerThread);

__global__ void __launch_bounds__(kTopKSelectThreads) dflash_topk16_chunk_select_kernel(
    const std::uint16_t* logits, int vocab, const float* chunk_max, const int* chunk_nan,
    std::int64_t* cand_ids, float* cand_unary) {
    constexpr int kInvalid = 0x7fffffff;
    __shared__ float warp_values[32];
    __shared__ int warp_ids[32];
    __shared__ int selected[kTopKChunkCandidates];
    __shared__ int selected_count;
    const int row = static_cast<int>(blockIdx.x);
    const int t = static_cast<int>(threadIdx.x);
    const int chunks = (vocab + kTopKChunk - 1) / kTopKChunk;
    const float* maxima = chunk_max + static_cast<std::size_t>(row) * kTopKMaxChunks;
    float values[kTopKSelectPerThread];
    int ids[kTopKSelectPerThread];
    bool nan = false;
    #pragma unroll
    for (int j = 0; j < kTopKSelectPerThread; ++j) {
        const int c = t + j * kTopKSelectThreads;
        values[j] = c < chunks ? maxima[c] : -CUDART_INF_F;
        ids[j] = c < chunks ? c : kInvalid;
        if (c < chunks) nan = nan || chunk_nan[static_cast<std::size_t>(row) * kTopKMaxChunks + c];
    }
    if (t == 0) selected_count = 0;
    if (__syncthreads_or(nan)) {
        if (t == 0) dflash_topk16_serial_row(logits, row, vocab, cand_ids, cand_unary);
        return;
    }
    // T = 16th largest chunk maximum (with multiplicity).
    float threshold = -CUDART_INF_F;
    for (int rank = 0; rank < kTopK; ++rank) {
        float value = -CUDART_INF_F;
        int id = kInvalid;
        #pragma unroll
        for (int j = 0; j < kTopKSelectPerThread; ++j)
            if (dflash_topk_precedes(values[j], ids[j], value, id)) { value = values[j]; id = ids[j]; }
        dflash_block_first(value, id, warp_values, warp_ids);
        threshold = value;
        #pragma unroll
        for (int j = 0; j < kTopKSelectPerThread; ++j)
            if (ids[j] == id) { values[j] = -CUDART_INF_F; ids[j] = kInvalid; }
    }
    for (int c = t; c < chunks; c += kTopKSelectThreads) {
        if (maxima[c] >= threshold) {
            const int slot = atomicAdd(&selected_count, 1);
            if (slot < kTopKChunkCandidates) selected[slot] = c;
        }
    }
    __syncthreads();
    const int count = selected_count;
    if (count > kTopKChunkCandidates) {
        if (t == 0) dflash_topk16_serial_row(logits, row, vocab, cand_ids, cand_unary);
        return;
    }
    const auto* input = logits + static_cast<std::size_t>(row) * vocab;
    #pragma unroll
    for (int j = 0; j < kTopKSelectPerThread; ++j) {
        const int i = t + j * kTopKSelectThreads;
        const int slot = i / kTopKChunk;
        const int column = slot < count ? selected[slot] * kTopKChunk + i % kTopKChunk : vocab;
        values[j] = column < vocab ? half_to_float(input[column]) : -CUDART_INF_F;
        ids[j] = column < vocab ? column : kInvalid;
    }
    for (int rank = 0; rank < kTopK; ++rank) {
        float value = -CUDART_INF_F;
        int id = kInvalid;
        #pragma unroll
        for (int j = 0; j < kTopKSelectPerThread; ++j)
            if (dflash_topk_precedes(values[j], ids[j], value, id)) { value = values[j]; id = ids[j]; }
        dflash_block_first(value, id, warp_values, warp_ids);
        if (t == 0) {
            cand_ids[static_cast<std::size_t>(row) * kTopK + rank] = id;
            cand_unary[static_cast<std::size_t>(row) * kTopK + rank] = value;
        }
        #pragma unroll
        for (int j = 0; j < kTopKSelectPerThread; ++j)
            if (ids[j] == id) { values[j] = -CUDART_INF_F; ids[j] = kInvalid; }
    }
}

bool dflash_topk_chunked_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_DFLASH2_TOPK_CHUNKED");
        return !value || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool dflash_topk_segmented_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_DFLASH2_TOPK_SEGMENTED");
        return !value || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

std::size_t dflash_topk_segment_bytes() {
    return std::max(static_cast<std::size_t>(kBlockCap) * kTopKSegments *
            (kTopK * (sizeof(float) + sizeof(int)) + sizeof(int)),
        static_cast<std::size_t>(kBlockCap) * kTopKMaxChunks * (sizeof(float) + sizeof(int)));
}

void launch_dflash_topk16_segmented(const std::uint16_t* logits, int rows, int vocab,
    std::int64_t* cand_ids, float* cand_unary, void* scratch, cudaStream_t stream) {
    if (dflash_topk_chunked_enabled()) {
        auto* chunk_max = static_cast<float*>(scratch);
        auto* chunk_nan = reinterpret_cast<int*>(chunk_max + kBlockCap * kTopKMaxChunks);
        const int chunks = (vocab + kTopKChunk - 1) / kTopKChunk;
        constexpr int warps = kTopKChunkThreads / 32;
        dflash_topk16_chunk_max_kernel<<<dim3((chunks + warps - 1) / warps, rows),
            kTopKChunkThreads, 0, stream>>>(logits, vocab, chunk_max, chunk_nan);
        dflash_topk16_chunk_select_kernel<<<rows, kTopKSelectThreads, 0, stream>>>(
            logits, vocab, chunk_max, chunk_nan, cand_ids, cand_unary);
        return;
    }
    auto* values = static_cast<float*>(scratch);
    auto* ids = reinterpret_cast<int*>(values + kBlockCap * kTopKSegments * kTopK);
    auto* nan = ids + kBlockCap * kTopKSegments * kTopK;
    dflash_topk16_segment_kernel<<<dim3(kTopKSegments, rows), kTopKSegmentThreads, 0, stream>>>(
        logits, vocab, values, ids, nan);
    dflash_topk16_segment_merge_kernel<<<rows, 32, 0, stream>>>(
        logits, vocab, values, ids, nan, cand_ids, cand_unary);
}

void dflash2_topk16_for_test(const std::uint16_t* logits, int rows, int vocab,
                           std::int64_t* ids, float* values, bool parallel,
                           bool local_merge,
                           cudaStream_t stream,
                           unsigned int* nonfinite_flag) {
    require(rows >= 0 && rows < kBlockCap && vocab >= kTopK && vocab <= kVocab + 256,
            "top-K test entry requires rows0..7 and vocab16..248576");
    if (rows == 0) return;
    require(logits && ids && values, "top-K test entry received null buffers");
    require(!nonfinite_flag || local_merge,
            "fused top-K liveness test requires local merge");
    if (local_merge && !nonfinite_flag && dflash_topk_segmented_enabled()) {
        static void* scratch = nullptr;
        if (!scratch)
            cuda_check(cudaMalloc(&scratch, dflash_topk_segment_bytes()),
                       "top-K segmented test scratch");
        launch_dflash_topk16_segmented(logits, rows, vocab, ids, values, scratch, stream);
    } else if (local_merge) {
        if (nonfinite_flag)
            dflash_topk16_local_merge_kernel<true><<<rows, 256, 0, stream>>>(
                logits, rows, vocab, ids, values, nonfinite_flag);
        else
            dflash_topk16_local_merge_kernel<false><<<rows, 256, 0, stream>>>(
                logits, rows, vocab, ids, values, nullptr);
    }
    else if (parallel)
        dflash_topk16_parallel_kernel<<<rows, 256, 0, stream>>>(logits, rows, vocab, ids, values);
    else
        dflash_topk16_kernel<<<rows, 1, 0, stream>>>(logits, rows, vocab, ids, values);
    cuda_check(cudaGetLastError(), "launch top-K qualification");
}

void dflash2_ring_attention_for_test(const std::uint16_t* q,
    const std::uint16_t* ring_k, const std::uint16_t* ring_v, int start, int count,
    const std::uint16_t* k, const std::uint16_t* v, std::uint16_t* out,
    int queries, int block, float scale, bool parallel, cudaStream_t stream) {
    require(start >= 0 && start < kRingCap && count >= 0 && count <= kRingKeep &&
            queries >= 0 && queries <= kBlockCap && block >= 1 && block <= kBlockCap &&
            std::isfinite(scale) && scale > 0, "ring attention test geometry");
    if (queries == 0) return;
    require(q && k && v && out && (count == 0 || (ring_k && ring_v)),
            "ring attention test null buffers");
    if (parallel && dflash_ring_split_selected(count)) {
        static float* partial_o = nullptr;
        static float* partial_ml = nullptr;
        if (!partial_o) {
            cuda_check(cudaMalloc(&partial_o, kRingSplitPartialFloats * sizeof(float)),
                       "ring split qualification partials");
            cuda_check(cudaMalloc(&partial_ml, kRingSplitMlFloats * sizeof(float)),
                       "ring split qualification max/sum");
        }
        launch_ring_attention_split(q, ring_k, ring_v, start, count, k, v, out, queries,
                                    block, scale, partial_o, partial_ml, stream);
    } else if (parallel && dflash_ring_staged_enabled())
        dflash_attention_ring_staged_kernel<<<queries * kQHeads, kHeadDim, 0, stream>>>(
            q, ring_k, ring_v, start, count, k, v, out, queries, block, scale);
    else if (parallel)
        dflash_attention_ring_parallel_kernel<<<queries * kQHeads, kHeadDim, 0, stream>>>(
            q, ring_k, ring_v, start, count, k, v, out, queries, block, scale);
    else
        dflash_attention_ring_kernel<<<queries * kQHeads, 1, 0, stream>>>(
            q, ring_k, ring_v, start, count, k, v, out, queries, block, scale);
    cuda_check(cudaGetLastError(), "launch ring attention qualification");
}

// E5A2 diagnostic: count non-finite elements (env-gated stage tracing).
__global__ void dflash_count_nonfinite_kernel(const std::uint16_t* buffer,
                                              int count, unsigned long long* out,
                                              int is_bf16) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = is_bf16 ? bf16_to_float(buffer[index]) : half_to_float(buffer[index]);
        const bool bad = (value != value) || (value == INFINITY) || (value == -INFINITY);
        if (bad) {
            atomicAdd(reinterpret_cast<unsigned long long*>(out), 1ULL);
        }
    }
}

// The default liveness guard materializes every head-logit row on the host.
// This separately labeled fast profile keeps the same fail-closed predicate
// while reducing the guarded result to one device flag. It produces no
// proposal or target-approval data.
__global__ void dflash_nonfinite_flag_kernel(const std::uint16_t* buffer,
                                             int count, unsigned int* out) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float value = half_to_float(buffer[index]);
    if ((value != value) || value == INFINITY || value == -INFINITY)
        atomicExch(out, 1u);
}

bool dflash2_ring_attention_split_for_test(int count) {
    return dflash_ring_split_selected(count);
}

void dflash2_dense_t_for_test(const std::uint16_t* input,
    const std::uint16_t* weights,std::uint16_t* output,int rows,int k,int n,
    bool weights_kmajor,cudaStream_t stream) {
    require(rows>=1 && rows<=kBlockCap && k>=1 && n>=1,
        "draft dense qualification geometry");
    require(input && weights && output,"draft dense qualification pointers");
    const dim3 grid((n+255)/256,rows);
    if(weights_kmajor && launch_dflash_dense_rows_kmajor(input,weights,output,rows,k,n,stream)) {
    } else if(weights_kmajor)
        dflash_dense_gemm_t_kernel<true><<<grid,256,
            static_cast<std::size_t>(k)*sizeof(std::uint16_t),stream>>>(
            input,weights,output,rows,k,n);
    else
        dflash_dense_gemm_t_kernel<false><<<grid,256,
            static_cast<std::size_t>(k)*sizeof(std::uint16_t),stream>>>(
            input,weights,output,rows,k,n);
    cuda_check(cudaGetLastError(),"draft dense qualification launch");
}

// Selector edge score for every candidate at one proposal position:
// score[c] = unary[row][c] + sum_r pred_cb[anchor][r] * hid[row][r] * succ_cb[cand][r]
__global__ void dflash_selector_edges_kernel(int position, std::int64_t anchor,
                                             const std::int64_t* device_anchor,
                                             const std::int64_t* cand_ids,
                                             const std::uint16_t* hidden,
                                             const std::uint16_t* pred_cb,
                                             const std::uint16_t* succ_cb,
                                             const float* cand_unary,
                                             float* scores) {
    const int c = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    extern __shared__ float shared[];
    const std::int64_t resolved_anchor=
        position==0 && device_anchor?device_anchor[0]:anchor;
    const std::uint16_t* pred_row = pred_cb + static_cast<std::size_t>(resolved_anchor) * kRank;
    const std::int64_t candidate =
        cand_ids[static_cast<std::size_t>(position) * kTopK + c];
    const std::uint16_t* succ_row =
        succ_cb + static_cast<std::size_t>(candidate) * kRank;
    const std::uint16_t* hid_row = hidden + static_cast<std::size_t>(position) * kRank;
    float acc = 0.0f;
    for (int r = lane; r < kRank; r += blockDim.x) {
        acc += half_to_float(pred_row[r]) * half_to_float(hid_row[r]) *
               half_to_float(succ_row[r]);
    }
    shared[lane] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) shared[lane] += shared[lane + stride];
        __syncthreads();
    }
    if (lane == 0) {
        scores[c] =
            cand_unary[static_cast<std::size_t>(position) * kTopK + c] + shared[0];
    }
}

// Default-off physical-C1 selector candidate. It retains the existing
// candidate order, FP32 reduction order, and strict-greater tie rule while
// keeping the bounded autoregressive chain on one block. The proposal output
// slot used by device-seed validity is intentionally untouched.
// One warp per candidate. Lane L holds the products of ranks L + 32 k and
// reproduces the 256-wide shared-memory tree (strides 128, 64, 32 in registers,
// then 16..1 by shuffles), so every score and the selected chain are those of
// the former one-product-per-thread reduction.
constexpr int kSelectorFusedThreads = kTopK * 32;
static_assert(kRank == 256, "selector tree is laid out for rank 256");

__global__ void __launch_bounds__(kSelectorFusedThreads) dflash_selector_fused_chain_kernel(
    int proposal_rows, std::int64_t initial_anchor,
    const std::int64_t* device_anchor, const std::int64_t* cand_ids,
    const std::uint16_t* hidden, const std::uint16_t* pred_cb,
    const std::uint16_t* succ_cb, const float* cand_unary,
    std::int64_t* proposal_out) {
    const int t = static_cast<int>(threadIdx.x);
    const int lane = t & 31;
    const int candidate_index = t >> 5;
    if (blockIdx.x != 0 || t >= kSelectorFusedThreads) return;
    __shared__ float scores[kTopK];
    __shared__ std::int64_t anchor;
    if (t == 0)
        anchor = device_anchor != nullptr ? device_anchor[0] : initial_anchor;
    __syncthreads();

    for (int position = 0; position < proposal_rows; ++position) {
        const auto* pred_row = pred_cb + static_cast<std::size_t>(anchor) * kRank;
        const auto* hid_row = hidden + static_cast<std::size_t>(position) * kRank;
        const auto candidate = cand_ids[
            static_cast<std::size_t>(position) * kTopK + candidate_index];
        const auto* succ_row = succ_cb + static_cast<std::size_t>(candidate) * kRank;
        float r[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int rank = lane + 32 * k;
            r[k] = half_to_float(pred_row[rank]) * half_to_float(hid_row[rank]) *
                   half_to_float(succ_row[rank]);
        }
        #pragma unroll
        for (int k = 0; k < 4; ++k) r[k] += r[k + 4];
        #pragma unroll
        for (int k = 0; k < 2; ++k) r[k] += r[k + 2];
        float value = r[0] + r[1];
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffu, value, offset);
        if (lane == 0)
            scores[candidate_index] = cand_unary[
                static_cast<std::size_t>(position) * kTopK + candidate_index] + value;
        __syncthreads();
        if (t == 0) {
            float best_score = scores[0];
            int best = 0;
            for (int c = 1; c < kTopK; ++c)
                if (scores[c] > best_score) { best_score = scores[c]; best = c; }
            const auto selected = cand_ids[
                static_cast<std::size_t>(position) * kTopK + best];
            proposal_out[position] = selected;
            anchor = selected;
        }
        __syncthreads();
    }
}

__global__ void dflash_prepare_device_seed_kernel(
    const Exl3GreedyRow* source,std::uint64_t expected_serial,
    std::int64_t mask_token,std::int64_t* ids,std::int64_t* status) {
    if(threadIdx.x || blockIdx.x)return;
    const auto row=source[0];
    const bool valid=row.serial==expected_serial && !row.nonfinite &&
        row.token>=0 && row.token<kVocab;
    ids[0]=valid?row.token:mask_token;
    *status=valid?0:1;
}

// argmax over the 16 scored candidates at one position -> selected token id.
__global__ void dflash_selector_argmax_kernel(int position, const float* scores,
                                              const std::int64_t* cand_ids,
                                              std::int64_t* proposal_out) {
    int best = 0;
    for (int c = 1; c < kTopK; ++c) {
        if (scores[c] > scores[best]) best = c;
    }
    proposal_out[position] =
        cand_ids[static_cast<std::size_t>(position) * kTopK + best];
}

// T73A test-only selector telemetry. The token tie-breaking is intentionally
// identical to dflash_selector_argmax_kernel. This kernel is never launched on
// the unset/default path.
__global__ void dflash_selector_argmax_confidence_kernel(
    int position, const float* scores, const std::int64_t* cand_ids,
    const float* cand_unary, std::int64_t* proposal_out,
    Exl3Dflash2DraftModel::DraftPositionConfidence* confidence_out) {
    int best = 0;
    int runner_up = 1;
    if (scores[runner_up] > scores[best]) {
        const int swap = best;
        best = runner_up;
        runner_up = swap;
    }
    for (int c = 2; c < kTopK; ++c) {
        if (scores[c] > scores[best]) {
            runner_up = best;
            best = c;
        } else if (scores[c] > scores[runner_up]) {
            runner_up = c;
        }
    }
    const std::size_t row = static_cast<std::size_t>(position) * kTopK;
    int unary_best = 0;
    int unary_runner_up = 1;
    if (cand_unary[row + unary_runner_up] > cand_unary[row + unary_best]) {
        const int swap = unary_best;
        unary_best = unary_runner_up;
        unary_runner_up = swap;
    }
    for (int c = 2; c < kTopK; ++c) {
        if (cand_unary[row + c] > cand_unary[row + unary_best]) {
            unary_runner_up = unary_best;
            unary_best = c;
        } else if (cand_unary[row + c] > cand_unary[row + unary_runner_up]) {
            unary_runner_up = c;
        }
    }
    proposal_out[position] = cand_ids[row + best];
    Exl3Dflash2DraftModel::DraftPositionConfidence out{};
    out.selected_edge_score = scores[best];
    out.runner_up_edge_score = scores[runner_up];
    out.edge_margin = scores[best] - scores[runner_up];
    out.selected_unary_score = cand_unary[row + best];
    out.best_unary_score = cand_unary[row + unary_best];
    out.runner_up_unary_score = cand_unary[row + unary_runner_up];
    out.unary_margin = out.best_unary_score - out.runner_up_unary_score;
    out.selected_candidate_rank = best;
    confidence_out[position] = out;
}

// Opt-in selector fast path.  The predecessor anchor is data-dependent, so
// the proposal positions cannot be launched as independent blocks.  This one
// persistent block keeps the 16 candidate scores resident while advancing the
// anchor chain on device, eliminating the per-position host copy/synchronize
// pair.  Candidate order and every argmax comparison intentionally remain the
// reference order; this kernel is never launched on the default path.
constexpr int kSelectorChainLanes = 64;
constexpr int kSelectorChainThreads = kTopK * kSelectorChainLanes;
__global__ void dflash_selector_batched_anchor_chain_kernel(
    int proposal_rows, std::int64_t initial_anchor,
    const std::int64_t* device_anchor, const std::int64_t* cand_ids,
    const std::uint16_t* hidden, const std::uint16_t* pred_cb,
    const std::uint16_t* succ_cb, const float* cand_unary, float* scores,
    std::int64_t* proposal_out,
    Exl3Dflash2DraftModel::DraftPositionConfidence* confidence_out) {
    // Keep one product slot per original edge-kernel lane.  The reduction
    // below deliberately mirrors its 128..1 tree, including the first level
    // where this 64-lane candidate group handles two pairs per lane.
    __shared__ float partial[kTopK * kRank];
    __shared__ std::int64_t chain_anchor;

    const int tid = static_cast<int>(threadIdx.x);
    const int candidate_rank = tid / kSelectorChainLanes;
    const int lane = tid % kSelectorChainLanes;
    if (tid == 0)
        chain_anchor = device_anchor ? device_anchor[0] : initial_anchor;
    __syncthreads();

    for (int position = 0; position < proposal_rows; ++position) {
        const std::size_t row = static_cast<std::size_t>(position) * kTopK;
        const std::int64_t candidate = cand_ids[row + candidate_rank];
        const std::uint16_t* pred_row =
            pred_cb + static_cast<std::size_t>(chain_anchor) * kRank;
        const std::uint16_t* succ_row =
            succ_cb + static_cast<std::size_t>(candidate) * kRank;
        const std::uint16_t* hid_row = hidden +
            static_cast<std::size_t>(position) * kRank;

        float* candidate_partial = partial + candidate_rank * kRank;
        for (int r = lane; r < kRank; r += kSelectorChainLanes) {
            candidate_partial[r] = half_to_float(pred_row[r]) *
                                  half_to_float(hid_row[r]) *
                                  half_to_float(succ_row[r]);
        }
        __syncthreads();
        for (int stride = kRank / 2; stride > 0; stride >>= 1) {
            if (stride > kSelectorChainLanes) {
                for (int r = lane; r < stride; r += kSelectorChainLanes)
                    candidate_partial[r] += candidate_partial[r + stride];
            } else if (lane < stride) {
                candidate_partial[lane] += candidate_partial[lane + stride];
            }
            __syncthreads();
        }
        if (lane == 0)
            scores[candidate_rank] = cand_unary[row + candidate_rank] +
                                     candidate_partial[0];
        __syncthreads();

        if (tid == 0) {
            int best = 0;
            int runner_up = 1;
            if (scores[runner_up] > scores[best]) {
                const int swap = best;
                best = runner_up;
                runner_up = swap;
            }
            for (int c = 2; c < kTopK; ++c) {
                if (scores[c] > scores[best]) {
                    runner_up = best;
                    best = c;
                } else if (scores[c] > scores[runner_up]) {
                    runner_up = c;
                }
            }
            proposal_out[position] = cand_ids[row + best];

            if (confidence_out != nullptr) {
                int unary_best = 0;
                int unary_runner_up = 1;
                if (cand_unary[row + unary_runner_up] >
                    cand_unary[row + unary_best]) {
                    const int swap = unary_best;
                    unary_best = unary_runner_up;
                    unary_runner_up = swap;
                }
                for (int c = 2; c < kTopK; ++c) {
                    if (cand_unary[row + c] > cand_unary[row + unary_best]) {
                        unary_runner_up = unary_best;
                        unary_best = c;
                    } else if (cand_unary[row + c] >
                               cand_unary[row + unary_runner_up]) {
                        unary_runner_up = c;
                    }
                }
                Exl3Dflash2DraftModel::DraftPositionConfidence out{};
                out.selected_edge_score = scores[best];
                out.runner_up_edge_score = scores[runner_up];
                out.edge_margin = scores[best] - scores[runner_up];
                out.selected_unary_score = cand_unary[row + best];
                out.best_unary_score = cand_unary[row + unary_best];
                out.runner_up_unary_score = cand_unary[row + unary_runner_up];
                out.unary_margin = out.best_unary_score -
                                   out.runner_up_unary_score;
                out.selected_candidate_rank = best;
                confidence_out[position] = out;
            }
            chain_anchor = proposal_out[position];
        }
        __syncthreads();
    }
}

void gopt_draft_rope_fixture(bool paired,const std::uint16_t* input,
    std::uint16_t* output,const int* positions,int rows,int heads,cudaStream_t stream) {
    if(paired)gopt_draft_rope_pair_kernel<<<(rows*heads*64+255)/256,256,0,stream>>>(input,output,positions,rows,heads);
    else dflash_rope_kernel<<<(rows*heads*128+255)/256,256,0,stream>>>(input,output,positions,rows,heads,128);
    cuda_check(cudaGetLastError(),"GOPT rotary fixture");
}
void gopt_draft_head_fixture(bool fused,bool paired,const std::uint16_t* input,
    const std::uint16_t* weight,std::uint16_t* normalized,std::uint16_t* rotated,
    const int* positions,int rows,int heads,cudaStream_t stream) {
    if(fused)dflash_head_norm_kernel<true><<<rows*heads,128,128*sizeof(float),stream>>>(
        input,weight,normalized,rows*heads,128,positions,rotated,heads,paired);
    else dflash_head_norm_kernel<false><<<rows*heads,128,128*sizeof(float),stream>>>(
        input,weight,normalized,rows*heads,128);
    cuda_check(cudaGetLastError(),"GOPT head fixture");
    if(!fused)gopt_draft_rope_fixture(paired,normalized,rotated,positions,rows,heads,stream);
}
void gopt_draft_ring_norm_fixture(bool fused,const std::uint16_t* k,
    const std::uint16_t* v,const std::uint16_t* weight,const int* positions,
    std::uint16_t* normalized,std::uint16_t* rotated,std::uint16_t* ring_k,
    std::uint16_t* ring_v,int rows,int first,cudaStream_t stream) {
    static_assert(kHeadDim==128 && kRopeDim==128);
    if(fused)gopt_draft_ring_norm_rope_kernel<<<rows*kKVHeads,128,128*sizeof(float),stream>>>(
        k,v,weight,positions,ring_k,ring_v,rows*kKVHeads);
    else {
        dflash_head_norm_kernel<<<rows*kKVHeads,128,128*sizeof(float),stream>>>(
            k,weight,normalized,rows*kKVHeads,kHeadDim);
        dflash_rope_kernel<<<(rows*kKVHeads*kHeadDim+255)/256,256,0,stream>>>(
            normalized,rotated,positions,rows,kKVHeads,kHeadDim);
        dflash_prefill_ring_scatter_kernel<<<(rows*kKVHeads*kHeadDim+255)/256,256,0,stream>>>(
            rotated,v,ring_k,ring_v,rows,first);
    }
    cuda_check(cudaGetLastError(),"GOPT ring norm/RoPE fixture");
}
void gopt_draft_conv_fixture(bool fused,const std::uint16_t* input,
    const std::uint16_t* dynamic,const std::uint16_t* base,std::uint16_t* conv,
    const std::uint16_t* residual,std::uint16_t* output,int rows,cudaStream_t stream) {
    if(fused)dflash_dyn_conv_kernel<DFlashFmt::BF16,true><<<dim3(rows, kHidden / 256), 256,0,stream>>>(
        input,dynamic,base,conv,rows,1,residual,output);
    else dflash_dyn_conv_kernel<DFlashFmt::BF16,false><<<dim3(rows, kHidden / 256), 256,0,stream>>>(
        input,dynamic,base,conv,rows,1);
    cuda_check(cudaGetLastError(),"GOPT draft conv fixture");
    if(!fused)dflash_residual_kernel<<<(rows*kHidden+255)/256,256,0,stream>>>(residual,conv,output,rows*kHidden);
    cuda_check(cudaGetLastError(),"GOPT draft residual fixture");
}
void gopt_draft_norm_fixture(bool fused,const std::uint16_t* left,
    const std::uint16_t* right,const std::uint16_t* weight,std::uint16_t* residual,
    std::uint16_t* normalized,int rows,cudaStream_t stream) {
    if(fused)dflash_rms_norm_kernel<DFlashFmt::BF16,true><<<rows,256,256*sizeof(float),stream>>>(
        left,weight,normalized,rows,kHidden,right,residual);
    else {
        dflash_residual_kernel<<<(rows*kHidden+255)/256,256,0,stream>>>(left,right,residual,rows*kHidden);
        dflash_rms_norm_kernel<DFlashFmt::BF16><<<rows,256,256*sizeof(float),stream>>>(residual,weight,normalized,rows,kHidden);
    }
    cuda_check(cudaGetLastError(),"GOPT residual norm fixture");
}
void gopt_draft_dense_fixture(bool paired,bool kmajor,const std::uint16_t* input,
    const std::uint16_t* weight,std::uint16_t* output,int rows,int k,int n,cudaStream_t stream) {
    if(paired && rows>=2 && rows<=8 && k<=kHidden) {
        if(kmajor)gopt_draft_dense_rowpair_kernel<true><<<dim3((n+255)/256,(rows+1)/2),256,2*k*sizeof(std::uint16_t),stream>>>(input,weight,output,rows,k,n);
        else gopt_draft_dense_rowpair_kernel<false><<<dim3((n+255)/256,(rows+1)/2),256,2*k*sizeof(std::uint16_t),stream>>>(input,weight,output,rows,k,n);
    } else {
        if(kmajor)dflash_dense_gemm_t_kernel<true><<<dim3((n+255)/256,rows),256,k*sizeof(std::uint16_t),stream>>>(input,weight,output,rows,k,n);
        else dflash_dense_gemm_t_kernel<false><<<dim3((n+255)/256,rows),256,k*sizeof(std::uint16_t),stream>>>(input,weight,output,rows,k,n);
    }
    cuda_check(cudaGetLastError(),"GOPT dense fixture");
}

struct Exl3Dflash2DraftModel::Impl {
    Exl3GamingOptions gaming=Exl3GamingOptions::from_environment();
    GoptSubmissions gaming_submissions{};
    int device=0;
    bool parallel_topk = false;
    bool local_merge_topk = false;
    std::uint64_t local_merge_topk_calls = 0;
    bool parallel_ring_attention = false;
    bool position_confidence = false;
    bool dense_kmajor = false;
    std::uint64_t dense_kmajor_launches = 0;
    bool selector_batched_anchor_chain = false;
    std::uint64_t selector_batched_anchor_chain_calls = 0;
    bool fast_device_liveness = false;
    bool fused_topk_liveness = false;
    std::uint64_t fused_topk_liveness_calls = 0;
    bool fused_selector = false;
    std::uint64_t fused_selector_calls = 0;
    std::uint64_t device_liveness_checks = 0;
    std::vector<DraftPositionConfidence> last_position_confidence;
    std::array<std::int32_t,kContextCap> host_context_positions{};
    std::array<std::int32_t,kBlockCap> host_block_positions{};
    std::uint64_t host_control_generation=0;
    bool host_control_active=false;
    RingAttentionObserver ring_attention_observer = nullptr;
    void* ring_attention_observer_user = nullptr;
    ProjectionObserver projection_observer = nullptr;
    bool projection_observer_gateup=false;
    Exl3DraftSharedQExecutor shared_q_executor;
    bool shared_block_kv=false;
    bool shared_block_o=false;
    bool shared_block_down=false;
    bool shared_block_gateup=false;
    Exl3ActivationLifetime shared_mlp_activation_lifetime;
    void* projection_observer_user = nullptr;
    enum class TimingCategory { Projection, OtherProjection, DenseMisc, Attention, H6, Selector };

    struct TimingSegment {
        TimingCategory category = TimingCategory::DenseMisc;
        std::string scope;
        int layer = -1;
        int rows = 0;
        Exl3CudaLinearMetadata metadata{};
        std::string dispatch;
        int begin_event = 0;
        int end_event = 0;
    };

    struct ProjectionTiming {
        static constexpr int kEventCapacity = 96;
        bool enabled = false;
        std::array<cudaEvent_t, kEventCapacity> events{};
        int marker_count = 0;
        std::uint64_t proposal_sequence = 0;
        double record_cpu_ms = 0.0;
        std::vector<TimingSegment> segments;

        ~ProjectionTiming() {
            for (cudaEvent_t event : events) {
                if (event != nullptr) cudaEventDestroy(event);
            }
        }

        void initialize() {
            const char* requested = std::getenv("NINFER_DFLASH2_PROJECTION_TIMING");
            enabled = requested != nullptr && std::string(requested) == "1";
            if (!enabled) return;
            for (auto& event : events)
                cuda_check(cudaEventCreate(&event), "P2 projection timing event create");
            segments.reserve(kEventCapacity - 1);
        }

        void record(cudaEvent_t event, cudaStream_t stream) {
            const auto begin = std::chrono::steady_clock::now();
            cuda_check(cudaEventRecord(event, stream), "P2 projection timing event record");
            record_cpu_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
        }

        void begin(cudaStream_t stream) {
            ++proposal_sequence;
            marker_count = 1;
            record_cpu_ms = 0.0;
            segments.clear();
            record(events[0], stream);
        }

        void close(TimingCategory category, std::string scope, int layer,
                   int rows, const Exl3CudaLinearMetadata* metadata,
                   const char* dispatch, cudaStream_t stream) {
            require(marker_count < kEventCapacity, "P2 projection timing event capacity exceeded");
            record(events[marker_count], stream);
            TimingSegment segment;
            segment.category = category;
            segment.scope = std::move(scope);
            segment.layer = layer;
            segment.rows = rows;
            if (metadata != nullptr) segment.metadata = *metadata;
            if (dispatch != nullptr) segment.dispatch = dispatch;
            segment.begin_event = marker_count - 1;
            segment.end_event = marker_count;
            segments.push_back(std::move(segment));
            ++marker_count;
        }
    } projection_timing;

    struct Layer {
        Exl3CudaLinearWeights q{}, k{}, v{}, o{}, gate{}, up{}, down{};
        Exl3CudaLinearMetadata qm{}, km{}, vm{}, om{}, gm{}, um{}, dm{};
        const std::uint16_t* input_norm = nullptr;
        const std::uint16_t* post_norm = nullptr;
        const std::uint16_t* q_norm = nullptr;
        const std::uint16_t* k_norm = nullptr;
        const std::uint16_t* attn_kernel_proj = nullptr;
        const std::uint16_t* mlp_kernel_proj = nullptr;
        const std::uint16_t* attn_base = nullptr;
        const std::uint16_t* mlp_base = nullptr;
    };
    std::array<Layer, kLayers> layers{};
    Exl3CudaLinearWeights fc{};
    Exl3CudaLinearMetadata fc_meta{};
    const std::uint16_t* hidden_norm = nullptr;
    const std::uint16_t* norm = nullptr;
    const std::uint16_t* hidden_proj = nullptr;
    const std::uint16_t* pred_cb = nullptr;
    const std::uint16_t* succ_cb = nullptr;

    std::filesystem::path directory;
    SafetensorsHeader header;
    // Published after load; subsequent execution resources retain this owner
    // without copying device payloads. No mutable request state lives here.
    std::shared_ptr<std::vector<std::unique_ptr<DeviceAllocation>>> allocations =
        std::make_shared<std::vector<std::unique_ptr<DeviceAllocation>>>();
    std::size_t weight_bytes = 0;

    bool prefill_batch = false;
    bool target_commit_batch = false;
    Exl3CudaLinearWorkspace::Owner ws_fc;
    Exl3CudaLinearWorkspace::Owner ws_q;
    Exl3CudaLinearWorkspace::Owner ws_kv;
    Exl3CudaLinearWorkspace::Owner ws_o;
    Exl3CudaLinearWorkspace::Owner ws_mlp;
    Exl3CudaLinearWorkspace::Owner ws_down;
    Exl3CudaLinearWorkspace::Owner ws_head;
    std::array<Exl3CudaLinearWorkspace*,7> linear_owners() const noexcept {
        return {ws_fc.get(),ws_q.get(),ws_kv.get(),ws_o.get(),ws_mlp.get(),ws_down.get(),ws_head.get()};
    }

    // E5A2 minimal native scratch. Block rows <= 8; context window rows <= 16.
    struct Scratch {
        const std::uint16_t** tap_ptr = nullptr;   // device array of 5 tap row pointers
        std::uint16_t* x_a = nullptr;       // [L][5120] layer state A (BF16 residual storage)
        std::uint16_t* x_b = nullptr;       // [L][5120] layer state B (BF16 residual storage)
        std::uint16_t* ln = nullptr;        // [L][5120]
        std::uint16_t* conv = nullptr;      // [L][5120]
        std::uint16_t* convf = nullptr;     // [L][5120] attention conv finish (BF16)
        std::uint16_t* dyn = nullptr;       // [L][1280]
        std::uint16_t* ln2 = nullptr;       // [L][5120]
        std::uint16_t* conv2 = nullptr;     // [L][5120]
        std::uint16_t* convf2 = nullptr;    // [L][5120] mlp conv finish (BF16)
        std::uint16_t* dyn2 = nullptr;      // [L][1280]
        std::uint16_t* qproj = nullptr;     // [L][4096]
        std::uint16_t* qn = nullptr;        // [L][4096]
        std::uint16_t* qr = nullptr;        // [L][4096]
        std::uint16_t* kblk = nullptr;      // [L][1024]
        std::uint16_t* vblk = nullptr;      // [L][1024]
        std::uint16_t* kblk_n = nullptr;    // [L][1024]
        std::uint16_t* kblk_r = nullptr;    // [L][1024]
        std::uint16_t* attn = nullptr;      // [L][4096]
        std::uint16_t* oproj = nullptr;     // [L][5120]
        std::uint16_t* gate = nullptr;      // [L][17408]
        std::uint16_t* up = nullptr;        // [L][17408]
        std::uint16_t* act = nullptr;       // [L][17408]
        std::uint16_t* down = nullptr;      // [L][5120]
        std::uint16_t* tcat = nullptr;      // [S][25600]
        std::uint16_t* fc_raw = nullptr;    // [S][5120]
        std::uint16_t* hctx = nullptr;      // [S][5120]
        std::uint16_t* kctx = nullptr;      // [S][1024]
        std::uint16_t* vctx = nullptr;      // [S][1024]
        std::uint16_t* kctx_n = nullptr;    // [S][1024]
        std::uint16_t* kctx_r = nullptr;    // [S][1024]
        std::uint16_t* final_norm = nullptr;    // [P][5120]
        std::uint16_t* head_out = nullptr;      // [P][248320]
        unsigned int* liveness_flag = nullptr;  // [1], FAST_DEVICE_LIVENESS only
        float* ring_partial_o = nullptr;        // split ring attention partials
        float* ring_partial_ml = nullptr;       // split ring attention max/sum
        std::uint16_t* hidden_proj_out = nullptr; // [P][256]
        std::int32_t* pos_ctx = nullptr;    // [S]
        std::int32_t* pos_blk = nullptr;    // [L]
        std::int64_t* ids = nullptr;        // [L]
        std::int64_t* cand_ids = nullptr;   // [P][16]
        float* cand_unary = nullptr;        // [P][16]
        void* topk_segments = nullptr;      // segmented top-K scratch
        float* edge_scores = nullptr;       // [16]
        std::int64_t* proposal_out = nullptr; // [P]
        DraftPositionConfidence* confidence_out = nullptr; // [P], T73A only
    } s{};

    std::vector<std::unique_ptr<DeviceAllocation>> scratch;
    std::size_t scratch_bytes = 0;

    // Draft KV allocations for E5A2 (small capacity, reported only). propose()
    // recomputes the windowed context each cycle; no cross-cycle persistence is used.
    std::array<std::uint16_t*, kLayers> kv_k{};
    std::array<std::uint16_t*, kLayers> kv_v{};
    std::size_t kv_capacity = 512;
    std::size_t kv_bytes = 0;
    // E5A3 bounded ring KV: per-layer K (post-k_norm, post-RoPE, absolute pos)
    // + V (raw), slot(p) = p & (kRingCap-1). Committed span is contiguous.
    std::array<std::uint16_t*, kLayers> ring_k{};
    std::array<std::uint16_t*, kLayers> ring_v{};
    // One device-only undo arena: five K/V pairs, at most eight physical slots.
    std::uint16_t* ring_undo_arena = nullptr;
    struct RingUndo {
        enum class Phase : std::uint8_t { Idle, Armed, InFlight, Applied };
        Phase phase = Phase::Idle;
        cudaStream_t stream = nullptr;
        int rows = 0;
        long long abs_pos0 = 0;
        long long base_abs = 0;
        int count = 0;
        std::shared_ptr<const Exl3DraftHostRing> host_parent;
        bool host_failed = false;
        std::uint64_t revision = 0, witness_revision = 0;
        std::uint64_t acquisition = 0, execution = 0;
        std::uint64_t witness_acquisition = 0, witness_execution = 0;
        bool witness_ready = false;
    } ring_undo;
    long long ring_base_abs = 0;
    int ring_count = 0;
    FreshPrefillStatus fresh_prefill;
    cudaStream_t fresh_prefill_stream = nullptr;
    std::size_t ring_bytes = 0;
    std::shared_ptr<const int> host_ring_identity=std::make_shared<const int>(0);
    std::shared_ptr<const Exl3DraftHostRing> host_ring_parent;
    bool host_ring_failed=false;
    bool fail_next_host_export_completion=false;
    std::uint64_t ring_revision=0,witness_revision=0;
    std::uint64_t ring_acquisition=0,ring_execution=0;
    std::uint64_t witness_acquisition=0,witness_execution=0;
    bool ring_witness_ready=false;
    RingRestoreStats ring_restore_stats;
    void invalidate_ring_witness() {
        ring_witness_ready=false;
        if(ring_revision==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("draft ring revision exhausted");
        ++ring_revision;
    }
    void copy_ring_undo_slots(bool restore, cudaStream_t stream) {
        constexpr std::size_t row_elements = static_cast<std::size_t>(kKVHeads) * kHeadDim;
        const int first_slot = static_cast<int>(ring_undo.abs_pos0 & kRingMask);
        for (int layer = 0; layer < kLayers; ++layer) {
            for (int plane = 0; plane < 2; ++plane) {
                auto* ring = plane == 0 ? ring_k[layer] : ring_v[layer];
                auto* saved = ring_undo_arena +
                    (static_cast<std::size_t>(layer) * 2 + plane) * 8 * row_elements;
                for (int offset = 0; offset < ring_undo.rows;) {
                    const int slot = (first_slot + offset) & kRingMask;
                    const int count = std::min(ring_undo.rows - offset, kRingCap - slot);
                    auto* ring_segment = ring + static_cast<std::size_t>(slot) * row_elements;
                    auto* saved_segment = saved + static_cast<std::size_t>(offset) * row_elements;
                    cuda_check(cudaMemcpyAsync(restore ? ring_segment : saved_segment,
                                              restore ? saved_segment : ring_segment,
                                              static_cast<std::size_t>(count) * row_elements *
                                                  sizeof(std::uint16_t),
                                              cudaMemcpyDeviceToDevice, stream),
                               restore ? "draft ring undo restore" : "draft ring undo snapshot");
                    offset += count;
                }
            }
        }
    }
    std::uint64_t* ring_digest_partial = nullptr;  // [kRingCap]

    int block_capacity = kBlockCap;
    int context_capacity = kContextCap;
    std::size_t required_execution_bytes=0;

    struct PrivateExecutionRequirement {
        std::size_t linear_bytes=0,storage_bytes=0,owner_count=0,total_bytes=0;
    };
    PrivateExecutionRequirement plan_private_execution() {
        PrivateExecutionRequirement required;
        required.linear_bytes=Exl3Dflash2DraftModel::linear_workspace_bytes_required();
        visit_scratch_allocations([&](std::size_t bytes,void**,const char*,std::size_t&){
            required.storage_bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(required.storage_bytes,bytes);
            ++required.owner_count;
        });
        required.total_bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(required.linear_bytes,required.storage_bytes);
        return required;
    }
    void allocate_execution(std::size_t expected_private_bytes=0,Exl3VeriCacheServingCoordinator* authority=nullptr) {
        const auto required=plan_private_execution();
        require(!expected_private_bytes || required.total_bytes==expected_private_bytes,
            "draft clone configuration changed private allocation requirement");
        required_execution_bytes=required.total_bytes;
        scratch.reserve(required.owner_count);
        const auto linear=[&](int input,int output,int rows,bool small=false,bool fc=false) {
            std::optional<RetainedDeviceLedger::Ticket> device;
            std::optional<RetainedDescriptorLedger::Ticket> metadata;
            if(authority) {
                const auto plan=Exl3LinearWorkspaceRequirements::derive(input,output,rows);
                auto credits=authority->reserve_constructor_credits(plan.owned_bytes,Exl3CudaLinearWorkspace::metadata_bytes());
                device.emplace(std::move(credits.device));metadata.emplace(std::move(credits.metadata));
            }
            return std::make_unique<Exl3CudaLinearWorkspace>(input,output,rows,small,false,false,false,false,false,fc,
                Exl3CudaAccumulationView{},Exl3CudaTransformView{},false,false,false,std::move(device),std::move(metadata));
        };
        ws_fc=linear(kLayers*kHidden,kHidden,kContextCap,false,true);
        ws_q=linear(kHidden,kQHeads*kHeadDim,kBlockCap,true);
        ws_kv=linear(kHidden,kKVHeads*kHeadDim,kContextCap,true);
        ws_o=linear(kQHeads*kHeadDim,kHidden,kBlockCap,true);
        ws_mlp=linear(kHidden,kIntermediate,kBlockCap);
        ws_down=linear(kIntermediate,kHidden,kBlockCap,true);
        ws_head=linear(kHidden,kVocab,kBlockCap);
        const auto actual_linear=ws_fc->workspace_bytes()+ws_q->workspace_bytes()+ws_kv->workspace_bytes()+
            ws_o->workspace_bytes()+ws_mlp->workspace_bytes()+ws_down->workspace_bytes()+ws_head->workspace_bytes();
        require(actual_linear==required.linear_bytes,"draft private linear allocation requirement mismatch");
        allocate_scratch(authority);
        require(scratch_bytes+kv_bytes+ring_bytes==required.storage_bytes && scratch.size()==required.owner_count,
            "draft private storage allocation requirement mismatch");
    }
    void allocate_scratch(Exl3VeriCacheServingCoordinator* authority=nullptr) {
        visit_scratch_allocations([&](std::size_t bytes,void** pointer,const char* label,std::size_t& total) {
            Exl3LinearWorkspaceRequirements::allocate_owned(scratch,total,pointer,bytes,
                [&]{
                    if(!authority)return std::make_unique<DeviceAllocation>(bytes,label);
                    auto credits=authority->reserve_constructor_credits(bytes,DeviceAllocation::metadata_bytes());
                    return std::make_unique<DeviceAllocation>(bytes,label,std::move(credits.device),std::move(credits.metadata));
                });
        });
    }
    template<class Allocate>
    void visit_scratch_allocations(Allocate&& allocate_counted) {
        auto alloc = [&](std::size_t bytes, void** ptr, const char* label) {
            allocate_counted(bytes,ptr,label,scratch_bytes);
        };
        const std::size_t two = sizeof(std::uint16_t);
        const std::size_t kv_elems = static_cast<std::size_t>(kKVHeads) * kHeadDim;
        const auto rowsB = [](int n) { return static_cast<std::size_t>(n); };
        alloc(kLayers * sizeof(void*), reinterpret_cast<void**>(&s.tap_ptr), "dflash tap ptr array");
        alloc(rowsB(kBlockCap) * kHidden * two, reinterpret_cast<void**>(&s.x_a), "dflash x_a");
        alloc(rowsB(kBlockCap) * kHidden * two, reinterpret_cast<void**>(&s.x_b), "dflash x_b");
        alloc(rowsB(kBlockCap) * kHidden * two, reinterpret_cast<void**>(&s.ln), "dflash ln");
        alloc(rowsB(kBlockCap) * kHidden * two, reinterpret_cast<void**>(&s.conv), "dflash conv");
        alloc(rowsB(kBlockCap) * kHidden * two, reinterpret_cast<void**>(&s.convf), "dflash convf");
        alloc(rowsB(kBlockCap) * kConvDynamic * two, reinterpret_cast<void**>(&s.dyn), "dflash dyn");
        alloc(rowsB(kBlockCap) * kHidden * two, reinterpret_cast<void**>(&s.ln2), "dflash ln2");
        alloc(rowsB(kBlockCap) * kHidden * two, reinterpret_cast<void**>(&s.conv2), "dflash conv2");
        alloc(rowsB(kBlockCap) * kHidden * two, reinterpret_cast<void**>(&s.convf2), "dflash convf2");
        alloc(rowsB(kBlockCap) * kConvDynamic * two, reinterpret_cast<void**>(&s.dyn2), "dflash dyn2");
        alloc(rowsB(kBlockCap) * kQHeads * kHeadDim * two, reinterpret_cast<void**>(&s.qproj), "dflash qproj");
        alloc(rowsB(kBlockCap) * kQHeads * kHeadDim * two, reinterpret_cast<void**>(&s.qn), "dflash qn");
        alloc(rowsB(kBlockCap) * kQHeads * kHeadDim * two, reinterpret_cast<void**>(&s.qr), "dflash qr");
        alloc(rowsB(kBlockCap) * kv_elems * two, reinterpret_cast<void**>(&s.kblk), "dflash kblk");
        alloc(rowsB(kBlockCap) * kv_elems * two, reinterpret_cast<void**>(&s.vblk), "dflash vblk");
        alloc(rowsB(kBlockCap) * kv_elems * two, reinterpret_cast<void**>(&s.kblk_n), "dflash kblk_n");
        alloc(rowsB(kBlockCap) * kv_elems * two, reinterpret_cast<void**>(&s.kblk_r), "dflash kblk_r");
        alloc(rowsB(kBlockCap) * kQHeads * kHeadDim * two, reinterpret_cast<void**>(&s.attn), "dflash attn");
        alloc(rowsB(kBlockCap) * kHidden * two, reinterpret_cast<void**>(&s.oproj), "dflash oproj");
        alloc(rowsB(kBlockCap) * kIntermediate * two, reinterpret_cast<void**>(&s.gate), "dflash gate");
        alloc(rowsB(kBlockCap) * kIntermediate * two, reinterpret_cast<void**>(&s.up), "dflash up");
        alloc(rowsB(kBlockCap) * kIntermediate * two, reinterpret_cast<void**>(&s.act), "dflash act");
        alloc(rowsB(kBlockCap) * kHidden * two, reinterpret_cast<void**>(&s.down), "dflash down");
        alloc(rowsB(kContextCap) * kLayers * kHidden * two, reinterpret_cast<void**>(&s.tcat), "dflash tap concat");
        alloc(rowsB(kContextCap) * kHidden * two, reinterpret_cast<void**>(&s.fc_raw), "dflash fc_raw");
        alloc(rowsB(kContextCap) * kHidden * two, reinterpret_cast<void**>(&s.hctx), "dflash hctx");
        alloc(rowsB(kContextCap) * kv_elems * two, reinterpret_cast<void**>(&s.kctx), "dflash kctx");
        alloc(rowsB(kContextCap) * kv_elems * two, reinterpret_cast<void**>(&s.vctx), "dflash vctx");
        alloc(rowsB(kContextCap) * kv_elems * two, reinterpret_cast<void**>(&s.kctx_n), "dflash kctx_n");
        alloc(rowsB(kContextCap) * kv_elems * two, reinterpret_cast<void**>(&s.kctx_r), "dflash kctx_r");
        alloc(rowsB(kBlockCap) * kHidden * two, reinterpret_cast<void**>(&s.final_norm), "dflash final norm");
        alloc(rowsB(kBlockCap) * kVocab * two, reinterpret_cast<void**>(&s.head_out), "dflash head out");
        if (fast_device_liveness)
            alloc(sizeof(unsigned int), reinterpret_cast<void**>(&s.liveness_flag),
                  "dflash fast liveness flag");
        alloc(kRingSplitPartialFloats * sizeof(float), reinterpret_cast<void**>(&s.ring_partial_o),
              "dflash ring split partials");
        alloc(kRingSplitMlFloats * sizeof(float), reinterpret_cast<void**>(&s.ring_partial_ml),
              "dflash ring split max/sum");
        alloc(rowsB(kBlockCap) * kRank * two, reinterpret_cast<void**>(&s.hidden_proj_out), "dflash hidden proj");
        alloc(rowsB(kContextCap) * sizeof(std::int32_t), reinterpret_cast<void**>(&s.pos_ctx), "dflash ctx positions");
        alloc(rowsB(kBlockCap) * sizeof(std::int32_t), reinterpret_cast<void**>(&s.pos_blk), "dflash block positions");
        alloc(rowsB(kBlockCap) * sizeof(std::int64_t), reinterpret_cast<void**>(&s.ids), "dflash token ids");
        alloc(rowsB(kBlockCap) * kTopK * sizeof(std::int64_t), reinterpret_cast<void**>(&s.cand_ids), "dflash cand ids");
        alloc(rowsB(kBlockCap) * kTopK * sizeof(float), reinterpret_cast<void**>(&s.cand_unary), "dflash cand unary");
        alloc(dflash_topk_segment_bytes(), &s.topk_segments, "dflash segmented top-K scratch");
        alloc(kTopK * sizeof(float), reinterpret_cast<void**>(&s.edge_scores), "dflash edge scores");
        alloc(rowsB(kBlockCap) * sizeof(std::int64_t), reinterpret_cast<void**>(&s.proposal_out), "dflash proposal out");
        if (position_confidence)
            alloc(rowsB(kBlockCap) * sizeof(DraftPositionConfidence),
                  reinterpret_cast<void**>(&s.confidence_out),
                  "dflash position confidence");
        for (int layer = 0; layer < kLayers; ++layer) {
            for (int slot = 0; slot < 2; ++slot) {
                auto& destination=slot==0?kv_k[layer]:kv_v[layer];
                allocate_counted(kv_capacity*kv_elems*two,reinterpret_cast<void**>(&destination),
                    "dflash kv cache",kv_bytes);
            }
        }
        for (int layer = 0; layer < kLayers; ++layer) {
            const auto bytes=static_cast<std::size_t>(kRingCap)*kv_elems*two;
            allocate_counted(bytes,reinterpret_cast<void**>(&ring_k[layer]),"dflash ring K",ring_bytes);
            allocate_counted(bytes,reinterpret_cast<void**>(&ring_v[layer]),"dflash ring V",ring_bytes);
        }
        allocate_counted(static_cast<std::size_t>(kLayers)*2*8*kv_elems*two,
            reinterpret_cast<void**>(&ring_undo_arena),"dflash ring undo arena",ring_bytes);
        {
            allocate_counted(static_cast<std::size_t>(kRingCap)*sizeof(std::uint64_t),
                reinterpret_cast<void**>(&ring_digest_partial),"dflash ring digest",ring_bytes);
        }
    }
};
Exl3Dflash2DraftModel::Exl3Dflash2DraftModel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

GoptSubmissions Exl3Dflash2DraftModel::gaming_submissions() const noexcept {
    return impl_?impl_->gaming_submissions:GoptSubmissions{};
}

Exl3Dflash2DraftModel::~Exl3Dflash2DraftModel() = default;

int Exl3Dflash2DraftModel::block_capacity() const noexcept { return impl_->block_capacity; }
int Exl3Dflash2DraftModel::spec_capacity() const noexcept { return impl_->block_capacity - 1; }
int Exl3Dflash2DraftModel::draft_layers() const noexcept { return kLayers; }
std::size_t Exl3Dflash2DraftModel::weight_bytes() const noexcept { return impl_->weight_bytes; }
std::uint64_t Exl3Dflash2DraftModel::dense_kmajor_launches() const noexcept {
    return impl_->dense_kmajor_launches;
}
std::uint64_t Exl3Dflash2DraftModel::selector_batched_anchor_chain_calls() const noexcept {
    return impl_->selector_batched_anchor_chain_calls;
}
std::uint64_t Exl3Dflash2DraftModel::weight_owner_metadata_bytes() const {
    Exl3ResourceInventory::Requirement requirement;
    using Domain=Exl3ResourceInventory::Domain;
    const auto& owners=*impl_->allocations;
    requirement.add(Domain::host_metadata,1,sizeof(owners));
    if(owners.capacity())requirement.add(Domain::host_metadata,owners.capacity(),sizeof(std::unique_ptr<DeviceAllocation>));
    if(owners.size())requirement.add(Domain::host_metadata,owners.size(),DeviceAllocation::metadata_bytes());
    return requirement.units[static_cast<unsigned>(Domain::host_metadata)];
}
std::size_t Exl3Dflash2DraftModel::kv_bytes() const noexcept { return impl_->kv_bytes; }
std::size_t Exl3Dflash2DraftModel::execution_bytes_required() const noexcept {return impl_->required_execution_bytes;}
std::size_t Exl3Dflash2DraftModel::linear_workspace_bytes_required() {
    std::size_t bytes=0;
    for(const auto shape:std::array<std::array<int,3>,7>{{
        {kLayers*kHidden,kHidden,kContextCap},{kHidden,kQHeads*kHeadDim,kBlockCap},
        {kHidden,kKVHeads*kHeadDim,kContextCap},{kQHeads*kHeadDim,kHidden,kBlockCap},
        {kHidden,kIntermediate,kBlockCap},{kIntermediate,kHidden,kBlockCap},
        {kHidden,kVocab,kBlockCap}}})
        bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(bytes,
            Exl3LinearWorkspaceRequirements::derive(shape[0],shape[1],shape[2]).owned_bytes);
    return bytes;
}
std::uint64_t Exl3Dflash2DraftModel::execution_owner_metadata_bytes() const {
    Exl3ResourceInventory::Requirement requirement;
    using Domain=Exl3ResourceInventory::Domain;
    const auto& owners=impl_->scratch;
    if(owners.capacity())requirement.add(Domain::host_metadata,owners.capacity(),sizeof(std::unique_ptr<DeviceAllocation>));
    if(owners.size())requirement.add(Domain::host_metadata,owners.size(),DeviceAllocation::metadata_bytes());
    for(const auto* workspace:{impl_->ws_fc.get(),impl_->ws_q.get(),impl_->ws_kv.get(),
        impl_->ws_o.get(),impl_->ws_mlp.get(),impl_->ws_down.get(),impl_->ws_head.get()})
        if(workspace)requirement.add(Domain::host_metadata,1,Exl3CudaLinearWorkspace::metadata_bytes());
    return requirement.units[static_cast<unsigned>(Domain::host_metadata)];
}
std::uint64_t Exl3Dflash2DraftModel::execution_owner_metadata_bytes_required() const {
    const auto plan=impl_->plan_private_execution();
    Exl3ResourceInventory::Requirement requirement;
    using Domain=Exl3ResourceInventory::Domain;
    requirement.add(Domain::host_metadata,plan.owner_count,sizeof(std::unique_ptr<DeviceAllocation>));
    requirement.add(Domain::host_metadata,plan.owner_count,DeviceAllocation::metadata_bytes());
    requirement.add(Domain::host_metadata,7,Exl3CudaLinearWorkspace::metadata_bytes());
    return requirement.units[static_cast<unsigned>(Domain::host_metadata)];
}
Exl3ResourceInventory Exl3Dflash2DraftModel::execution_resources(
    const std::shared_ptr<Exl3Dflash2DraftModel>& owner,unsigned device_short,unsigned metadata_short) const {
    if(owner.get()!=this || !execution_bytes() || device_short>1 || metadata_short>1)
        throw std::invalid_argument("draft execution inventory owner/extent");
    std::uint64_t linear_bytes=0,linear_metadata=0;
    for(const auto* workspace:impl_->linear_owners())if(workspace) {
        linear_bytes+=workspace->workspace_bytes();linear_metadata+=Exl3CudaLinearWorkspace::metadata_bytes();
    }
    const auto bytes=execution_bytes(),metadata=execution_owner_metadata_bytes();
    if(bytes<=linear_bytes+device_short || metadata<=linear_metadata+metadata_short)
        throw std::logic_error("draft execution inventory missing scratch extent");
    using Domain=Exl3ResourceInventory::Domain;
    Exl3ResourceInventory actual;
    std::uint64_t generic_bytes=0,generic_metadata=0,slot=6;
    for(const auto& child:impl_->scratch) {
        if(!child || !child->bytes)throw std::logic_error("draft generic inventory missing allocation");
        generic_bytes+=child->bytes;generic_metadata+=DeviceAllocation::metadata_bytes();
        std::shared_ptr<const void> allocation(owner,child.get());
        const auto declared_bytes=child->bytes-(slot==6?device_short:0);
        actual.add({allocation,slot++,Domain::device,declared_bytes,{},nullptr,
            &DeviceAllocation::attach_device_credit});
        actual.add({allocation,slot++,Domain::host_metadata,DeviceAllocation::metadata_bytes(),{},
            &DeviceAllocation::attach_metadata_credit});
    }
    if(generic_bytes!=bytes-linear_bytes)throw std::logic_error("draft generic allocation inventory extent mismatch");
    actual.add({owner,3,Domain::host_metadata,metadata-linear_metadata-generic_metadata-metadata_short});
    actual.add({owner,4,Domain::host_metadata,linear_metadata,{},&attach_linear_metadata_credit});
    actual.add({owner,5,Domain::device,linear_bytes,{},nullptr,&attach_linear_device_credit});
    return actual;
}
bool Exl3Dflash2DraftModel::attach_linear_metadata_credit(const std::shared_ptr<const void>& owner,
    RetainedDescriptorLedger::Ticket credit) noexcept {
    if(!owner)return false;
    auto* draft=const_cast<Exl3Dflash2DraftModel*>(static_cast<const Exl3Dflash2DraftModel*>(owner.get()));
    const auto workspaces=draft->impl_->linear_owners();std::uint64_t bytes=0;
    for(const auto* workspace:workspaces) {
        if(!workspace || !workspace->can_attach_metadata_credit())return false;
        bytes+=Exl3CudaLinearWorkspace::metadata_bytes();
    }
    if(credit.bytes()!=bytes)return false;
    for(auto* workspace:workspaces) {
        auto part=credit.split(Exl3CudaLinearWorkspace::metadata_bytes());
        if(!part || !workspace->attach_metadata_credit(std::move(*part)))return false;
    }
    return true;
}
bool Exl3Dflash2DraftModel::attach_linear_device_credit(const std::shared_ptr<const void>& owner,
    RetainedDeviceLedger::Ticket credit) noexcept {
    if(!owner)return false;
    auto* draft=const_cast<Exl3Dflash2DraftModel*>(static_cast<const Exl3Dflash2DraftModel*>(owner.get()));
    const auto workspaces=draft->impl_->linear_owners();std::uint64_t bytes=0;
    for(const auto* workspace:workspaces) {
        if(!workspace || !workspace->can_attach_device_credit())return false;
        bytes+=workspace->workspace_bytes();
    }
    if(credit.bytes()!=bytes)return false;
    for(auto* workspace:workspaces) {
        auto part=credit.split(workspace->workspace_bytes());
        if(!part || !workspace->attach_device_credit(std::move(*part)))return false;
    }
    return true;
}
bool Exl3Dflash2DraftModel::fail_linear_retirement_for_test(unsigned index,bool after_accumulation) noexcept {
    const auto workspaces=impl_->linear_owners();
    if(index>=workspaces.size() || !workspaces[index])return false;
    workspaces[index]->fail_owned_retirement_for_test(after_accumulation);return true;
}
std::size_t Exl3Dflash2DraftModel::scratch_bytes() const noexcept { return impl_->scratch_bytes; }
std::uint64_t Exl3Dflash2DraftModel::generic_owner_metadata_bytes() const noexcept {
    return impl_->scratch.size()*DeviceAllocation::metadata_bytes();
}
std::uint64_t Exl3Dflash2DraftModel::generic_quarantined_allocations() noexcept {
    return DeviceAllocation::quarantined_count.load()+Exl3DraftHostRing::uncertain_transfer_count();
}
std::size_t Exl3Dflash2DraftModel::uncertain_source_allocations_for_test() const noexcept {
    std::size_t count=0;for(const auto& allocation:impl_->scratch)count+=allocation->uncertain_transfer_use;
    return count;
}
std::weak_ptr<const void> Exl3Dflash2DraftModel::uncertain_source_owner_for_test() const noexcept {
    for(const auto& allocation:impl_->scratch)
        if(allocation->uncertain_transfer_use)return allocation->source_lifetime;
    return {};
}
bool Exl3Dflash2DraftModel::fail_generic_retirement_for_test() noexcept {
    if(impl_->scratch.empty() || !impl_->scratch.front())return false;
    impl_->scratch.front()->cleanup_failure_for_test=true;return true;
}
Exl3Dflash2DraftModel::GenericRetirementSnapshot Exl3Dflash2DraftModel::latest_generic_retirement_for_test() noexcept {
    GenericRetirementSnapshot result;
    const auto* retained=DeviceAllocation::quarantine.load();
    if(!retained)return result;
    result.bytes=retained->bytes;result.device=retained->device;result.error=retained->error;
    result.pointer_retained=retained->pointer!=nullptr;result.record_bytes=sizeof(DeviceAllocation::Retained);
    result.device_credit=retained->device_credit?retained->device_credit->bytes():0;
    result.metadata_credit=retained->metadata_credit?retained->metadata_credit->bytes():0;
    return result;
}
void Exl3Dflash2DraftModel::exercise_generic_retirement_for_test(bool uncertain_transfer) {
    const auto before=generic_quarantined_allocations();
    RetainedDeviceLedger device;RetainedDescriptorLedger metadata;
    {
        auto control=std::make_shared<DeviceAllocation>(128,"draft generic retirement control");
        for(const auto bytes:{std::uint64_t(0),std::uint64_t(127),std::uint64_t(129)})
            require(!DeviceAllocation::attach_device_credit(control,device.acquire(bytes)) && device.bytes()==0,
                "draft generic invalid device credit retained charge");
        for(const auto bytes:{std::size_t(0),DeviceAllocation::metadata_bytes()-1,DeviceAllocation::metadata_bytes()+1})
            require(!DeviceAllocation::attach_metadata_credit(control,metadata.acquire(bytes)) && metadata.bytes()==0,
                "draft generic invalid metadata credit retained charge");
        require(DeviceAllocation::attach_device_credit(control,device.acquire(128)) &&
            DeviceAllocation::attach_metadata_credit(control,metadata.acquire(DeviceAllocation::metadata_bytes())),
            "draft generic exact credit attachment refused");
        require(!DeviceAllocation::attach_device_credit(control,device.acquire(128)) && device.bytes()==128 &&
            !DeviceAllocation::attach_metadata_credit(control,metadata.acquire(DeviceAllocation::metadata_bytes())) &&
            metadata.bytes()==DeviceAllocation::metadata_bytes(),"draft generic duplicate attachment changed original credits");
    }
    require(generic_quarantined_allocations()==before,"draft generic normal cleanup quarantined");
    require(device.bytes()==0 && metadata.bytes()==0,"draft generic successful cleanup retained tickets");
    const auto cleanup_before=DeviceAllocation::native_cleanup_attempts.load();
    const void* expected_pointer=nullptr;
    {
        auto failed=std::make_shared<DeviceAllocation>(128,"draft generic retirement failure");
        require(DeviceAllocation::attach_device_credit(failed,device.acquire(128)) &&
            DeviceAllocation::attach_metadata_credit(failed,metadata.acquire(DeviceAllocation::metadata_bytes())),
            "draft generic failed owner missing tickets");
        expected_pointer=failed->ptr;
        failed->cleanup_failure_for_test=!uncertain_transfer;
        failed->uncertain_transfer_use=uncertain_transfer;
    }
    const auto* retained=DeviceAllocation::quarantine.load(std::memory_order_acquire);
    require(generic_quarantined_allocations()==before+1 && retained && retained->pointer==expected_pointer && retained->bytes==128 &&
        retained->error==static_cast<int>(cudaErrorUnknown),"draft generic cleanup lost failed allocation");
    require(DeviceAllocation::native_cleanup_attempts.load()==cleanup_before,
        "uncertain or injected-failure source attempted native cleanup");
    require(device.bytes()==128 && metadata.bytes()==sizeof(DeviceAllocation::Retained) &&
        retained->device_credit && retained->device_credit->bytes()==128 &&
        retained->metadata_credit && retained->metadata_credit->bytes()==sizeof(DeviceAllocation::Retained),
        "draft generic quarantine lost exact device/record tickets or retained dead wrapper charge");
    bool refused=false;try{DeviceAllocation retry(128,"draft generic forbidden retry");}
    catch(const std::runtime_error&){refused=true;}
    require(refused && generic_quarantined_allocations()==before+1,"draft generic cleanup retried or admitted after quarantine");
    require(device.bytes()==128 && metadata.bytes()==sizeof(DeviceAllocation::Retained),
        "refused draft allocation changed quarantined lifetime credits");
}
std::size_t Exl3Dflash2DraftModel::execution_bytes() const noexcept {
    const auto& m=*impl_;
    if(!m.ws_fc)return 0; // Engine-only deferred execution, before startup credit.
    return m.scratch_bytes+m.kv_bytes+m.ring_bytes+m.ws_fc->workspace_bytes()+
        m.ws_q->workspace_bytes()+m.ws_kv->workspace_bytes()+m.ws_o->workspace_bytes()+
        m.ws_mlp->workspace_bytes()+m.ws_down->workspace_bytes()+m.ws_head->workspace_bytes();
}
bool Exl3Dflash2DraftModel::shares_weights_with(const Exl3Dflash2DraftModel& other) const noexcept {
    return impl_->allocations==other.impl_->allocations;
}
std::unique_ptr<Exl3Dflash2DraftModel> Exl3Dflash2DraftModel::create_execution() const {
    return create_execution_impl(false);
}
void Exl3Dflash2DraftModel::materialize_private_execution(Exl3VeriCacheServingCoordinator* authority) {
    std::unique_lock lock(execution_mutex_,std::try_to_lock);
    require(lock.owns_lock() && !impl_->ws_fc,"draft private construction requires deferred idle clone");
    int device=0;cuda_check(cudaGetDevice(&device),"draft materialization device");
    require(device==impl_->device,"draft materialization/weight device mismatch");
    impl_->projection_timing.initialize();
    impl_->allocate_execution(impl_->required_execution_bytes,authority);
}
void Exl3Dflash2DraftModel::finish_constructor_credits() noexcept {
    for(const auto& child:impl_->scratch)child->release_constructor_credits_after_commit();
    for(auto* child:impl_->linear_owners())if(child)child->release_constructor_credits_after_commit();
}
std::unique_ptr<Exl3Dflash2DraftModel> Exl3Dflash2DraftModel::create_execution_impl(bool defer_private) const {
    std::unique_lock lock(execution_mutex_,std::try_to_lock);
    require(lock.owns_lock(),"cannot clone an acquired draft execution resource");
    const auto& parent=*impl_;int device=0;cuda_check(cudaGetDevice(&device),"draft execution device");
    require(device==parent.device,"draft execution/weight device mismatch");
    auto child=std::make_unique<Impl>();child->device=device;
    child->allocations=parent.allocations;child->weight_bytes=parent.weight_bytes;
    child->directory=parent.directory;child->header=parent.header;
    child->layers=parent.layers;child->fc=parent.fc;child->fc_meta=parent.fc_meta;
    child->hidden_norm=parent.hidden_norm;child->norm=parent.norm;child->hidden_proj=parent.hidden_proj;
    child->pred_cb=parent.pred_cb;child->succ_cb=parent.succ_cb;
    child->host_ring_identity=parent.host_ring_identity;
    child->prefill_batch=parent.prefill_batch;child->target_commit_batch=parent.target_commit_batch;
    child->parallel_topk=parent.parallel_topk;
    child->local_merge_topk=parent.local_merge_topk;child->parallel_ring_attention=parent.parallel_ring_attention;
    child->dense_kmajor=parent.dense_kmajor;
    child->selector_batched_anchor_chain=parent.selector_batched_anchor_chain;
    child->fast_device_liveness=parent.fast_device_liveness;
    child->fused_topk_liveness=parent.fused_topk_liveness;
    child->fused_selector=parent.fused_selector;
    child->position_confidence=parent.position_confidence;
    child->required_execution_bytes=parent.required_execution_bytes;
    if(!defer_private) {
        child->projection_timing.initialize();child->allocate_execution(parent.required_execution_bytes);
    }
    return std::unique_ptr<Exl3Dflash2DraftModel>(new Exl3Dflash2DraftModel(std::move(child)));
}
const std::uint16_t* Exl3Dflash2DraftModel::last_head_input_device_for_test() const noexcept {
    return impl_->s.final_norm;
}

const std::uint16_t* Exl3Dflash2DraftModel::last_head_logits_device_for_test() const noexcept {
    return impl_->s.head_out;
}
const std::int64_t* Exl3Dflash2DraftModel::last_topk_ids_device_for_test() const noexcept {
    return impl_->s.cand_ids;
}
const float* Exl3Dflash2DraftModel::last_topk_values_device_for_test() const noexcept {
    return impl_->s.cand_unary;
}
std::uint64_t Exl3Dflash2DraftModel::local_topk_calls() const noexcept {
    return impl_->local_merge_topk_calls;
}
std::uint64_t Exl3Dflash2DraftModel::fused_topk_liveness_calls() const noexcept {
    return impl_->fused_topk_liveness_calls;
}
const std::vector<Exl3Dflash2DraftModel::DraftPositionConfidence>&
Exl3Dflash2DraftModel::last_position_confidence_for_test() const noexcept {
    return impl_->last_position_confidence;
}
Exl3Dflash2DraftModel::HostControlStorageSnapshot
Exl3Dflash2DraftModel::host_control_storage_for_test() const noexcept {
    return {reinterpret_cast<std::uintptr_t>(impl_->host_context_positions.data()),
        reinterpret_cast<std::uintptr_t>(impl_->host_block_positions.data()),
        impl_->host_context_positions.size(),impl_->host_block_positions.size(),
        impl_->host_control_generation,impl_->host_control_active};
}
void Exl3Dflash2DraftModel::require_host_control_idle_for_test() const {
    require(!impl_->host_control_active,"reentrant draft host control mutation");
}

void Exl3Dflash2DraftModel::set_shared_q_executor(Exl3DraftSharedQExecutor executor,bool block_kv,bool block_o,bool block_down,bool block_gateup) {
    std::unique_lock lock(execution_mutex_,std::try_to_lock);
    require(lock.owns_lock() && impl_->ring_acquisition==0,
        "draft shared Q installation requires unacquired execution");
    require_no_fresh_prefill();impl_->shared_q_executor=std::move(executor);
    impl_->shared_block_kv=block_kv;
    impl_->shared_block_o=block_o;
    impl_->shared_block_down=block_down;
    impl_->shared_block_gateup=block_gateup;
}

void Exl3Dflash2DraftModel::set_projection_observer_for_test(
    ProjectionObserver observer, void* user,bool include_gateup) noexcept {
    impl_->projection_observer = observer;
    impl_->projection_observer_user = observer != nullptr ? user : nullptr;
    impl_->projection_observer_gateup=observer && include_gateup;
}

void Exl3Dflash2DraftModel::set_ring_attention_observer_for_test(
    RingAttentionObserver observer, void* user) noexcept {
    impl_->ring_attention_observer = observer;
    impl_->ring_attention_observer_user = observer ? user : nullptr;
}

void Exl3Dflash2DraftModel::reset(cudaStream_t stream) {
    Exl3DraftHostRing::require_transfer_healthy();
    if (impl_->ring_undo.phase != Impl::RingUndo::Phase::Idle &&
        stream != impl_->ring_undo.stream) {
        impl_->host_ring_failed = true;
        require(false, "draft ring undo reset stream mismatch");
    }
    impl_->ring_undo = {};
    impl_->invalidate_ring_witness();
    if (impl_->fresh_prefill.active && stream != impl_->fresh_prefill_stream) {
        impl_->fresh_prefill.failed = true;
        require(false, "fresh prefill reset stream mismatch");
    }
    impl_->fresh_prefill = {};
    impl_->fresh_prefill_stream = nullptr;
    // E5A3: the ring holds only committed K/V addressed by absolute position, so
    // reset is host-side logical state (slots are overwritten by slot(p) mapping).
    impl_->ring_base_abs = 0;
    impl_->ring_count = 0;
    impl_->host_ring_parent.reset();impl_->host_ring_failed=false;
}

long long Exl3Dflash2DraftModel::ring_base_abs() const noexcept { return impl_->ring_base_abs; }

int Exl3Dflash2DraftModel::ring_count() const noexcept { return impl_->ring_count; }

std::size_t Exl3Dflash2DraftModel::ring_bytes() const noexcept { return impl_->ring_bytes; }

std::shared_ptr<const Exl3DraftHostRing> Exl3DraftHostRing::detached_payload_for_test() const {
    auto copy=Exl3DraftHostRing::create();
    copy->base_=base_;copy->count_=count_;
    copy->pages_.reserve(pages_.size());
    for(const auto& page:pages_)copy->pages_.push_back(make_bounded_shared<Page>(*page));
    return copy;
}
bool Exl3DraftHostRing::same_payload(const Exl3DraftHostRing& other) const {
    return model_identity_==other.model_identity_ && same_represented_payload_for_test(other);
}
bool Exl3DraftHostRing::same_represented_payload_for_test(const Exl3DraftHostRing& other) const {
    if(base_!=other.base_ || count_!=other.count_) return false;
    std::size_t a=0,b=0;
    for(long long row=base_;row<position();++row) {
        while(a<pages_.size() && pages_[a]->first+pages_[a]->rows<=row) ++a;
        while(b<other.pages_.size() && other.pages_[b]->first+other.pages_[b]->rows<=row) ++b;
        if(a==pages_.size() || b==other.pages_.size() || pages_[a]->first>row || other.pages_[b]->first>row) return false;
        for(int layer=0;layer<5;++layer) {
            const auto offset_a=static_cast<std::size_t>(row-pages_[a]->first)*1024;
            const auto offset_b=static_cast<std::size_t>(row-other.pages_[b]->first)*1024;
            if(std::memcmp(pages_[a]->k[layer].data()+offset_a,other.pages_[b]->k[layer].data()+offset_b,2048) ||
                std::memcmp(pages_[a]->v[layer].data()+offset_a,other.pages_[b]->v[layer].data()+offset_b,2048)) return false;
        }
    }
    return true;
}

std::size_t Exl3DraftHostRing::write_represented_payload_for_test(
    const std::filesystem::path& path) const {
    std::ofstream out(path,std::ios::binary | std::ios::trunc);
    if(!out)throw std::runtime_error("draft represented ring file open failed");
    constexpr std::uint64_t magic=0x31474e4952443545ULL;
    out.write(reinterpret_cast<const char*>(&magic),sizeof(magic));
    out.write(reinterpret_cast<const char*>(&base_),sizeof(base_));
    out.write(reinterpret_cast<const char*>(&count_),sizeof(count_));
    std::size_t bytes=sizeof(magic)+sizeof(base_)+sizeof(count_);
    std::size_t page_index=0;
    for(long long row=base_;row<position();++row) {
        while(page_index<pages_.size() &&
              pages_[page_index]->first+pages_[page_index]->rows<=row)
            ++page_index;
        if(page_index==pages_.size() || pages_[page_index]->first>row)
            throw std::logic_error("draft represented ring page gap");
        const auto& page=*pages_[page_index];
        const auto offset=static_cast<std::size_t>(row-page.first)*1024;
        for(int layer=0;layer<5;++layer) {
            for(const auto* plane:{&page.k[layer],&page.v[layer]}) {
                if(plane->size()<offset+1024)
                    throw std::logic_error("draft represented ring plane extent");
                out.write(reinterpret_cast<const char*>(plane->data()+offset),2048);
                bytes+=2048;
            }
        }
    }
    out.flush();
    if(!out)throw std::runtime_error("draft represented ring file write failed");
    return bytes;
}

std::uint64_t Exl3DraftHostRing::visit_allocations(
    std::span<const std::shared_ptr<const Exl3DraftHostRing>> states,
    const std::function<void(const void*,std::size_t)>& visitor,bool include_unused_capacity) {
    std::unordered_set<const Page*> seen;std::uint64_t bytes=0;
    for(const auto& state:states) {
        require(state!=nullptr,"draft host accounting null state");
        for(const auto& page:state->pages_) if(seen.insert(page.get()).second) {
            const auto visit=[&](const auto& plane) {
                const auto extent=(include_unused_capacity?plane.capacity():plane.size())*2;
                bytes+=extent;if(visitor && extent) visitor(plane.data(),extent);
            };
            for(const auto& plane:page->k) visit(plane);
            for(const auto& plane:page->v) visit(plane);
        }
    }
    return bytes;
}

void Exl3Dflash2DraftModel::fail_next_host_export_completion_for_test() {
    require_no_fresh_prefill();
    impl_->fail_next_host_export_completion=true;
}
std::shared_ptr<const Exl3DraftHostRing> Exl3Dflash2DraftModel::export_host_ring(cudaStream_t stream,bool share_prefix,
    const Exl3DraftHostRing::MetadataReservation& reserve,bool fail_completion_for_test) {
    require_no_fresh_prefill();auto& m=*impl_;
    m.ring_witness_ready=false;
    require(!m.host_ring_failed && m.ring_count>0 && m.ring_count<=kRingCap,
        "draft host export requires a completed bounded nonempty ring");
    require(Exl3DraftHostRing::uncertain_transfer_count()==0,"unresolved draft host export transfer");
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    cuda_check(cudaStreamIsCapturing(stream,&capture),"draft host export capture status");
    require(capture==cudaStreamCaptureStatusNone,"draft host export requires eager stream");
    cuda_check(cudaStreamSynchronize(stream),"draft host export completed ring");
    auto result=Exl3DraftHostRing::create_planned(Exl3DraftHostRing::planned_page_capacity(m.ring_base_abs,m.ring_count),reserve);
    result->model_identity_=m.host_ring_identity;result->base_=m.ring_base_abs;result->count_=m.ring_count;
    const auto end=result->position();
    const auto parent=share_prefix?m.host_ring_parent:nullptr;
    long long cursor=result->base_;
    if(parent) {
        require(parent->model_identity_==m.host_ring_identity && parent->base_<=result->base_ && parent->position()<=end,
            "draft host append lineage mismatch");
        for(const auto& page:parent->pages_) if(page->first+page->rows>result->base_) {
            require(result->pages_.size()<result->pages_.capacity(),"draft ring parent exceeds planned descriptors");
            result->pages_.push_back(page);
        }
        cursor=std::max(cursor,parent->position());
    }
    struct DownloadPage {
        std::shared_ptr<Exl3DraftHostRing::Page> page;
        long long cursor=0;
        int previous=0,rows=0;
    };
    std::array<DownloadPage,kRingCap/64+1> downloads{};
    std::size_t download_count=0;
    while(cursor<end) {
        auto page=Exl3DraftHostRing::create_page(reserve);
        if(!result->pages_.empty() && result->pages_.back()->first+result->pages_.back()->rows==cursor && cursor%64) {
            *page=*result->pages_.back();result->pages_.pop_back();
        } else page->first=cursor;
        const int rows=static_cast<int>(std::min<long long>(end-cursor,64-cursor%64));
        const int previous=page->rows;page->rows+=rows;
        for(int layer=0;layer<kLayers;++layer) {
            page->k[layer].reserve(64*1024);page->v[layer].reserve(64*1024);
            page->k[layer].resize(static_cast<std::size_t>(page->rows)*1024);
            page->v[layer].resize(static_cast<std::size_t>(page->rows)*1024);
        }
        require(download_count<downloads.size(),"draft ring download plan capacity");
        downloads[download_count++]={page,cursor,previous,rows};
        require(result->pages_.size()<result->pages_.capacity(),"draft ring export exceeds planned descriptors");
        result->pages_.push_back(std::move(page));cursor+=rows;
    }
    // All metadata and payload storage is prepared before the first download.
    // No partial preparation can publish a new parent or submit an earlier page.
    for(std::size_t index=0;index<download_count;++index) {
        const auto& planned=downloads[index];const auto& page=planned.page;
        for(int layer=0;layer<kLayers;++layer) {
            const auto download=[&](auto& plane,const std::uint16_t* source) {
                const auto retain_uncertain=[&] {
                    // Keep source allocations out of cudaFree as well as keeping
                    // destination storage alive; retirement records are preallocated.
                    for(const auto& allocation:m.scratch)
                        if(allocation->ptr==source)allocation->uncertain_transfer_use=true;
                    m.host_ring_failed=true;Exl3DraftHostRing::retain_uncertain_transfer(result);
                };
                const auto submitted=cudaMemcpyAsync(plane.data()+static_cast<std::size_t>(planned.previous)*1024,
                    source+static_cast<std::size_t>(planned.cursor&kRingMask)*1024,static_cast<std::size_t>(planned.rows)*2048,
                    cudaMemcpyDeviceToHost,stream);
                if(submitted!=cudaSuccess) {
                    retain_uncertain();
                    cuda_check(submitted,"draft host page suffix download");
                }
                const bool fail_completion=std::exchange(fail_completion_for_test,false) |
                    std::exchange(m.fail_next_host_export_completion,false);
                const auto completed=fail_completion?cudaErrorUnknown:cudaStreamSynchronize(stream);
                if(completed!=cudaSuccess) {
                    retain_uncertain();
                    cuda_check(completed,"draft host suffix complete");
                }
                result->exported_bytes_+=static_cast<std::size_t>(planned.rows)*2048;
            };
            download(page->k[layer],m.ring_k[layer]);download(page->v[layer],m.ring_v[layer]);
        }
    }
    m.host_ring_parent=result;m.witness_revision=m.ring_revision;m.ring_witness_ready=true;
    m.witness_acquisition=m.ring_acquisition;m.witness_execution=m.ring_execution;
    return result;
}

void Exl3Dflash2DraftModel::bind_ring_scope(std::uint64_t acquisition,std::uint64_t execution) {
    Exl3DraftHostRing::require_transfer_healthy();
    require(impl_->ring_undo.phase == Impl::RingUndo::Phase::Idle,
            "draft ring undo must finish before scope binding");
    impl_->invalidate_ring_witness();
    impl_->ring_acquisition=acquisition;impl_->ring_execution=execution;
}
bool Exl3Dflash2DraftModel::rebind_ring_scope_if_resident(
    const std::shared_ptr<const Exl3DraftHostRing>& state,
    std::uint64_t acquisition,std::uint64_t execution) {
    Exl3DraftHostRing::require_transfer_healthy();
    auto& m=*impl_;
    if (m.ring_undo.phase != Impl::RingUndo::Phase::Idle) return false;
    if(!acquisition || !execution || !m.ring_acquisition || !m.ring_execution ||
       !host_ring_resident(state))return false;
    m.ring_acquisition=acquisition;m.ring_execution=execution;
    m.witness_acquisition=acquisition;m.witness_execution=execution;
    return true;
}
bool Exl3Dflash2DraftModel::host_ring_resident(const std::shared_ptr<const Exl3DraftHostRing>& state) const noexcept {
    const auto& m=*impl_;
    // Strong immutable parent ownership precludes same-address replacement.
    // Every ring writer invalidates the content revision before submission;
    // exports/restores bind it only after successful stream completion.
    return Exl3DraftHostRing::uncertain_transfer_count()==0 && state && !m.fresh_prefill.active &&
        m.ring_undo.phase == Impl::RingUndo::Phase::Idle && !m.host_ring_failed &&
        m.ring_witness_ready && m.witness_revision==m.ring_revision &&
        m.witness_acquisition==m.ring_acquisition && m.witness_execution==m.ring_execution &&
        state==m.host_ring_parent && state->model_identity_==m.host_ring_identity &&
        state->base_==m.ring_base_abs && state->count_==m.ring_count;
}
bool Exl3Dflash2DraftModel::restore_host_ring_if_needed(
    std::shared_ptr<const Exl3DraftHostRing> state,cudaStream_t stream) {
    if(host_ring_resident(state)) {++impl_->ring_restore_stats.skipped;return false;}
    restore_host_ring(std::move(state),stream);return true;
}
bool Exl3Dflash2DraftModel::host_ring_lineage(
    const std::shared_ptr<const Exl3DraftHostRing>& state,
    std::uint64_t acquisition,std::uint64_t execution) const noexcept {
    return acquisition!=0 && execution!=0 && impl_->ring_acquisition==acquisition &&
        impl_->ring_execution==execution && host_ring_resident(state);
}
std::shared_ptr<const Exl3DraftHostRing> Exl3Dflash2DraftModel::completed_ring_parent_for_test() const noexcept {
    const auto& parent=impl_->host_ring_parent;
    return host_ring_lineage(parent,impl_->ring_acquisition,impl_->ring_execution)?parent:nullptr;
}
Exl3Dflash2DraftModel::RingRestoreStats Exl3Dflash2DraftModel::ring_restore_stats() const noexcept {
    return impl_->ring_restore_stats;
}

void Exl3Dflash2DraftModel::restore_host_ring(std::shared_ptr<const Exl3DraftHostRing> state,cudaStream_t stream) {
    require_no_fresh_prefill();auto& m=*impl_;
    m.invalidate_ring_witness();++m.ring_restore_stats.required;
    require(state && state->model_identity_==m.host_ring_identity && state->count_>0 && state->count_<=kRingKeep &&
        state->base_>=0 && state->position()<=2147483647LL,"draft host restore identity/extent mismatch");
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    cuda_check(cudaStreamIsCapturing(stream,&capture),"draft host restore capture status");
    require(capture==cudaStreamCaptureStatusNone,"draft host restore requires eager stream");
    m.host_ring_parent.reset();m.host_ring_failed=true;
    try {
        for(const auto& page:state->pages_) {
            const auto first=std::max(state->base_,page->first);
            const auto end=std::min(state->position(),page->first+page->rows);
            if(end<=first) continue;
            for(int layer=0;layer<kLayers;++layer) {
                const auto upload=[&](std::uint16_t* destination,const auto& plane) {
                    cuda_check(cudaMemcpyAsync(destination+static_cast<std::size_t>(first&kRingMask)*1024,
                        plane.data()+static_cast<std::size_t>(first-page->first)*1024,static_cast<std::size_t>(end-first)*2048,
                        cudaMemcpyHostToDevice,stream),"draft host page restore");
                    cuda_check(cudaStreamSynchronize(stream),"draft host restore complete");
                };
                upload(m.ring_k[layer],page->k[layer]);upload(m.ring_v[layer],page->v[layer]);
            }
        }
        m.ring_base_abs=state->base_;m.ring_count=state->count_;m.host_ring_parent=std::move(state);m.host_ring_failed=false;
        m.witness_revision=m.ring_revision;m.ring_witness_ready=true;
        m.witness_acquisition=m.ring_acquisition;m.witness_execution=m.ring_execution;
    } catch(...) {m.ring_base_abs=0;m.ring_count=0;throw;}
}

void Exl3Dflash2DraftModel::rewind_to(int count, cudaStream_t) {
    require_no_fresh_prefill();
    impl_->invalidate_ring_witness();
    // E5A3 logical rollback (mechanism A): no copies, no snapshots. Slots beyond
    // the rewound length are simply reused by future commits (slot = pos mod cap,
    // and the committed span never exceeds 2047 < 2048, so no aliasing).
    require(count >= 0 && count <= impl_->ring_count,
            "E5A3 rewind target outside committed ring length");
    impl_->ring_count = count;
    impl_->host_ring_parent.reset();
}

std::array<std::uint64_t, 5> Exl3Dflash2DraftModel::ring_digest(cudaStream_t stream) {
    Exl3DraftHostRing::require_transfer_healthy();
    Impl& m = *impl_;
    require(m.ring_undo.phase == Impl::RingUndo::Phase::Idle,
            "draft ring digest requires a settled undo scope");
    std::array<std::uint64_t, 5> out{};
    const int start_slot = static_cast<int>(m.ring_base_abs & kRingMask);
    for (int layer = 0; layer < kLayers; ++layer) {
        if (m.ring_count == 0) { out[static_cast<std::size_t>(layer)] = 1469598103934665603ULL; continue; }
        dflash_ring_digest_kernel<<<(m.ring_count + 255) / 256, 256, 0, stream>>>(
            m.ring_k[layer], m.ring_v[layer], start_slot, m.ring_count,
            m.ring_digest_partial);
        cuda_check(cudaGetLastError(), "E5A3 ring digest launch");
        std::vector<std::uint64_t> partial(static_cast<std::size_t>(m.ring_count));
        cuda_check(cudaMemcpyAsync(partial.data(), m.ring_digest_partial,
                                   partial.size() * sizeof(std::uint64_t),
                                   cudaMemcpyDeviceToHost, stream),
                   "E5A3 ring digest download");
        cuda_check(cudaStreamSynchronize(stream), "E5A3 ring digest sync");
        std::uint64_t h = 1469598103934665603ULL;
        for (const auto v : partial) { h ^= v + 0x9e3779b97f4a7c15ULL; h *= 1099511628211ULL; }
        out[static_cast<std::size_t>(layer)] = h;
    }
    return out;
}

std::array<std::uint64_t, 5> Exl3Dflash2DraftModel::physical_ring_digest_for_test(
    cudaStream_t stream) {
    Exl3DraftHostRing::require_transfer_healthy();
    Impl& m = *impl_;
    require(m.ring_undo.phase == Impl::RingUndo::Phase::Idle,
            "physical ring digest requires a settled undo scope");
    std::array<std::uint64_t, 5> out{};
    for (int layer = 0; layer < kLayers; ++layer) {
        dflash_ring_digest_kernel<<<(kRingCap + 255) / 256, 256, 0, stream>>>(
            m.ring_k[layer], m.ring_v[layer], 0, kRingCap,
            m.ring_digest_partial);
        cuda_check(cudaGetLastError(), "physical ring digest launch");
        std::vector<std::uint64_t> partial(kRingCap);
        cuda_check(cudaMemcpyAsync(partial.data(), m.ring_digest_partial,
                                   partial.size() * sizeof(std::uint64_t),
                                   cudaMemcpyDeviceToHost, stream),
                   "physical ring digest download");
        cuda_check(cudaStreamSynchronize(stream), "physical ring digest sync");
        std::uint64_t h = 1469598103934665603ULL;
        for (const auto v : partial) {
            h ^= v + 0x9e3779b97f4a7c15ULL;
            h *= 1099511628211ULL;
        }
        out[static_cast<std::size_t>(layer)] = h;
    }
    return out;
}

void Exl3Dflash2DraftModel::require_no_fresh_prefill() {
    Exl3DraftHostRing::require_transfer_healthy();
    require(impl_->ring_undo.phase == Impl::RingUndo::Phase::Idle,
            "draft ring undo must finish before ordinary operation");
    if (impl_->fresh_prefill.active || impl_->fresh_prefill.failed) {
        impl_->fresh_prefill.failed = true;
        require(false, "fresh prefill must finish or reset before ordinary operation");
    }
}

Exl3Dflash2DraftModel::FreshPrefillStatus
Exl3Dflash2DraftModel::fresh_prefill_status() const noexcept { return impl_->fresh_prefill; }

void Exl3Dflash2DraftModel::begin_fresh_prefill(long long start, long long end,
                                             cudaStream_t stream) {
    require_no_fresh_prefill();
    require(impl_->ring_count == 0, "fresh prefill requires empty logical ring");
    require(start >= 0 && end > start && end <= 2147483647LL,
            "fresh prefill interval out of range");
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cuda_check(cudaStreamIsCapturing(stream, &capture), "fresh prefill capture status");
    require(capture == cudaStreamCaptureStatusNone, "fresh prefill requires eager stream");
    impl_->fresh_prefill = {};
    auto& f = impl_->fresh_prefill;
    f.active = true; f.start = start; f.end = end; f.submitted_cursor = start;
    impl_->fresh_prefill_stream = stream;
}

void Exl3Dflash2DraftModel::skip_fresh_prefill_block(int rows,
    long long abs_pos0, cudaStream_t stream) {
    require(impl_->ring_undo.phase == Impl::RingUndo::Phase::Idle,
            "draft ring undo must finish before fresh prefill skip");
    auto& f = impl_->fresh_prefill;
    try {
        Exl3DraftHostRing::require_transfer_healthy();
        require(f.active && !f.failed, "fresh prefill skip requires active scope");
        require(stream == impl_->fresh_prefill_stream,
                "fresh prefill skip stream mismatch");
        require(rows >= 1 && rows <= 16 && abs_pos0 == f.submitted_cursor &&
                abs_pos0 >= f.start && abs_pos0 + rows <= f.end - kRingCap,
                "fresh prefill skip requires an ordered whole discarded call");
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        cuda_check(cudaStreamIsCapturing(stream, &capture),
                   "fresh prefill skip capture status");
        require(capture == cudaStreamCaptureStatusNone,
                "fresh prefill skip requires eager stream");
        f.submitted_rows += rows;
        f.skipped_rows += rows;
        f.submitted_cursor += rows;
    } catch (...) { f.failed = true; throw; }
}

void Exl3Dflash2DraftModel::finish_fresh_prefill(cudaStream_t stream) {
    Exl3DraftHostRing::require_transfer_healthy();
    require(impl_->ring_undo.phase == Impl::RingUndo::Phase::Idle,
            "draft ring undo must finish before fresh prefill finish");
    auto& f = impl_->fresh_prefill;
    try {
        require(f.active && !f.failed, "fresh prefill scope unavailable");
        require(stream == impl_->fresh_prefill_stream, "fresh prefill finish stream mismatch");
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        cuda_check(cudaStreamIsCapturing(stream, &capture), "fresh prefill finish capture status");
        require(capture == cudaStreamCaptureStatusNone, "fresh prefill finish requires eager stream");
        require(f.submitted_cursor == f.end && f.submitted_rows == f.end - f.start &&
                f.encoded_rows + f.skipped_rows == f.submitted_rows,
                "fresh prefill submitted coverage incomplete");
        const auto count = std::min<long long>(kRingKeep, f.end - f.start);
        require(impl_->ring_count == count && impl_->ring_base_abs == f.end - count,
                "fresh prefill final ring span mismatch");
        f.active = false;
    } catch (...) { f.failed = true; throw; }
}

void Exl3Dflash2DraftModel::commit_target_block(const std::uint16_t* const* taps,
    int rows, long long abs_pos0, cudaStream_t stream) {
    require_no_fresh_prefill();
    try {
        // Target settlement is opt-in and bounded to the existing exact
        // prefill batch geometry. Keep one-row and larger-than-eight commits
        // on the original serial path.
        if (impl_->target_commit_batch && impl_->prefill_batch && rows >= 2 && rows <= 8)
            commit_prefill_block_internal(taps, rows, abs_pos0, stream);
        else
            commit_target_block_internal(taps, rows, abs_pos0, stream);
    }
    catch(...) {impl_->host_ring_failed=true;impl_->host_ring_parent.reset();throw;}
}

void Exl3Dflash2DraftModel::begin_prefill_ring_undo(int rows, long long abs_pos0,
    cudaStream_t stream) {
    require_no_fresh_prefill();
    auto& m = *impl_;
    require(m.ring_undo.phase == Impl::RingUndo::Phase::Idle,
            "draft ring undo already active");
    require(!m.host_ring_failed && m.ring_undo_arena && m.ring_count == kRingKeep,
            "draft ring undo requires a healthy saturated ring");
    require(m.ring_acquisition != 0 && m.ring_execution != 0,
            "draft ring undo requires a bound request scope");
    require(rows >= 1 && rows <= 8 && abs_pos0 == m.ring_base_abs + m.ring_count &&
            abs_pos0 >= 0 && abs_pos0 <= 2147483647LL - rows,
            "draft ring undo extent must be one contiguous 1..8-row commit");
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cuda_check(cudaStreamIsCapturing(stream, &capture), "draft ring undo capture status");
    require(capture == cudaStreamCaptureStatusNone, "draft ring undo requires eager stream");
    auto& undo = m.ring_undo;
    undo.phase = Impl::RingUndo::Phase::Armed;
    undo.stream = stream;
    undo.rows = rows;
    undo.abs_pos0 = abs_pos0;
    undo.base_abs = m.ring_base_abs;
    undo.count = m.ring_count;
    undo.host_parent = m.host_ring_parent;
    undo.host_failed = m.host_ring_failed;
    undo.revision = m.ring_revision;
    undo.witness_revision = m.witness_revision;
    undo.acquisition = m.ring_acquisition;
    undo.execution = m.ring_execution;
    undo.witness_acquisition = m.witness_acquisition;
    undo.witness_execution = m.witness_execution;
    undo.witness_ready = m.ring_witness_ready;
    try { m.copy_ring_undo_slots(false, stream); }
    catch (...) {
        m.ring_undo = {};
        m.host_ring_failed = true;
        m.host_ring_parent.reset();
        m.ring_count = 0;
        throw;
    }
}

void Exl3Dflash2DraftModel::accept_prefill_ring_undo(cudaStream_t stream) {
    auto& m = *impl_;
    require(m.ring_undo.phase == Impl::RingUndo::Phase::Applied &&
            stream == m.ring_undo.stream && !m.host_ring_failed &&
            m.ring_acquisition == m.ring_undo.acquisition &&
            m.ring_execution == m.ring_undo.execution,
            "draft ring undo accept requires one completed matching commit");
    try { cuda_check(cudaStreamSynchronize(stream), "draft ring undo commit completion"); }
    catch (...) {
        m.ring_undo = {};
        m.host_ring_failed = true;
        m.host_ring_parent.reset();
        m.ring_count = 0;
        throw;
    }
    m.ring_undo = {};
}

void Exl3Dflash2DraftModel::rollback_prefill_ring_undo(cudaStream_t stream) {
    auto& m = *impl_;
    require(m.ring_undo.phase != Impl::RingUndo::Phase::Idle &&
            stream == m.ring_undo.stream &&
            m.ring_acquisition == m.ring_undo.acquisition &&
            m.ring_execution == m.ring_undo.execution,
            "draft ring undo rollback requires its original stream");
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cuda_check(cudaStreamIsCapturing(stream, &capture), "draft ring undo rollback capture status");
    require(capture == cudaStreamCaptureStatusNone,
            "draft ring undo rollback requires eager stream");
    try {
        // Snapshot, forward projections, and restore are ordered on the same
        // stream. Completion must precede restoring the host-side ring witness.
        m.copy_ring_undo_slots(true, stream);
        cuda_check(cudaStreamSynchronize(stream), "draft ring undo rollback completion");
        auto& undo = m.ring_undo;
        m.ring_base_abs = undo.base_abs;
        m.ring_count = undo.count;
        m.host_ring_parent = std::move(undo.host_parent);
        m.host_ring_failed = undo.host_failed;
        m.ring_revision = undo.revision;
        m.witness_revision = undo.witness_revision;
        m.ring_acquisition = undo.acquisition;
        m.ring_execution = undo.execution;
        m.witness_acquisition = undo.witness_acquisition;
        m.witness_execution = undo.witness_execution;
        m.ring_witness_ready = undo.witness_ready;
        undo = {};
    } catch (...) {
        m.ring_undo = {};
        m.host_ring_failed = true;
        m.host_ring_parent.reset();
        m.ring_count = 0;
        throw;
    }
}

void Exl3Dflash2DraftModel::commit_prefill_block(const std::uint16_t* const* taps,
    int rows, long long abs_pos0, cudaStream_t stream) {
    Exl3DraftHostRing::require_transfer_healthy();
    auto& m = *impl_; auto& f = m.fresh_prefill;
    if (m.ring_undo.phase != Impl::RingUndo::Phase::Idle) {
        require(m.ring_undo.phase == Impl::RingUndo::Phase::Armed && !f.active &&
                !f.failed && stream == m.ring_undo.stream && rows == m.ring_undo.rows &&
                abs_pos0 == m.ring_undo.abs_pos0,
                "draft ring undo commit must match its armed interval and stream");
        m.ring_undo.phase = Impl::RingUndo::Phase::InFlight;
        try {
            commit_prefill_block_internal(taps, rows, abs_pos0, stream);
            m.ring_undo.phase = Impl::RingUndo::Phase::Applied;
        } catch (...) {
            m.host_ring_failed = true;
            m.host_ring_parent.reset();
            throw;
        }
        return;
    }
    if (!f.active) {
        require_no_fresh_prefill();
        try {commit_prefill_block_internal(taps, rows, abs_pos0, stream);}
        catch(...) {m.host_ring_failed=true;m.host_ring_parent.reset();throw;}
        return;
    }
    try {
        require(!f.failed, "fresh prefill scope failed; reset required");
        require(stream == m.fresh_prefill_stream, "fresh prefill commit stream mismatch");
        require(taps && rows >= 1 && rows <= 16, "draft prefill commit requires1..16 rows");
        for (int t = 0; t < kLayers; ++t) require(taps[t] != nullptr, "draft prefill tap pointer missing");
        require(abs_pos0 >= 0 && abs_pos0 <= 2147483647LL - rows, "draft prefill position out of range");
        require(abs_pos0 == f.submitted_cursor && abs_pos0 + rows <= f.end,
                "fresh prefill commit order or extent mismatch");
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        cuda_check(cudaStreamIsCapturing(stream, &capture), "draft prefill capture status");
        require(capture == cudaStreamCaptureStatusNone, "draft prefill commit requires eager stream");
        if (m.prefill_batch && rows > 1) {
            require(m.ws_fc->draft_prefill_fc_candidate(m.fc_meta, std::min(rows,8),
                    Exl3CudaLinearAdmission::draft_prefill_fc), "draft prefill FC admission unavailable");
            for (const auto& layer : m.layers)
                require(m.ws_kv->draft_prefill_kv_candidate(layer.km, std::min(rows,8)) &&
                        m.ws_kv->draft_prefill_kv_candidate(layer.vm, std::min(rows,8)),
                        "draft prefill K/V admission unavailable");
        }
        if (abs_pos0 + rows <= f.end - kRingCap) f.skipped_rows += rows;
        else {
            commit_prefill_block_internal(taps, rows, abs_pos0, stream);
            f.encoded_rows += rows;
        }
        f.submitted_rows += rows; f.submitted_cursor += rows;
    } catch (...) { f.failed = true; throw; }
}

void Exl3Dflash2DraftModel::commit_target_block_internal(const std::uint16_t* const* taps,
                                                int rows, long long abs_pos0,
                                                cudaStream_t stream) {
    impl_->invalidate_ring_witness();
    // E5A3 commit path: NEW target tap rows enter the ring through the identical
    // per-row pipeline the reference applies (tap concat -> fc -> hidden_norm ->
    // per-layer k/v proj -> k_norm -> RoPE at ABSOLUTE position -> ring scatter).
    // Matches reference update_and_fetch + offset discipline; bulk ingest keeps only
    // the newest kRingKeep rows (reference skip logic). Staging reuses row 0 of the
    // existing per-call scratch (tcat/fc_raw/hctx/kctx/vctx/kctx_n/kctx_r/pos_ctx).
    Impl& m = *impl_;
    require(taps != nullptr, "E5A3 commit needs tap row pointers");
    require(rows >= 1, "E5A3 commit row count must be positive");
    const long long end = m.ring_base_abs + m.ring_count;
    require(abs_pos0 == end || m.ring_count == 0,
            "E5A3 commit must extend the committed span contiguously");
    // Reference prefill-skip (S > sliding_window - 1 keeps the newest rows) applies
    // ONLY to a fresh prefill on an empty ring. On a non-empty ring every new row is
    // ingested and the per-row eviction below keeps the newest 2047 exactly, because
    // bulk rows are all newer than the committed span (matches RotatingKVCache trim).
    int first = 0;
    if (m.ring_count == 0 && rows > kRingKeep)
        first = rows - kRingKeep;  // reference skip: drop oldest bulk rows
    const long long first_abs = abs_pos0 + first;
    if (m.ring_count == 0) m.ring_base_abs = first_abs;
    const int kept = rows - first;
    const std::uint16_t* ptrs[kLayers];
    for (int r = 0; r < kept; ++r) {
        const long long pos = first_abs + r;
        for (int t = 0; t < kLayers; ++t)
            ptrs[t] = taps[t] + static_cast<std::size_t>(first + r) * kHidden;
        cuda_check(cudaMemcpyAsync(m.s.tap_ptr, ptrs, sizeof(ptrs),
                                   cudaMemcpyHostToDevice, stream),
                   "E5A3 commit tap pointers");
        dflash_tap_concat_kernel<<<(kLayers * kHidden + 255) / 256, 256, 0, stream>>>(m.s.tap_ptr, m.s.tcat, 1);
        cuda_check(cudaGetLastError(), "E5A3 commit tap concat");
        m.ws_fc->forward(m.fc, m.fc_meta, m.s.tcat, m.s.fc_raw, 1, stream);
        dflash_rms_norm_kernel<DFlashFmt::F16><<<1, 256, 256 * sizeof(float), stream>>>(
            m.s.fc_raw, m.hidden_norm, m.s.hctx, 1, kHidden);
        cuda_check(cudaGetLastError(), "E5A3 commit hidden norm");
        const std::int32_t p32 = static_cast<std::int32_t>(pos);
        cuda_check(cudaMemcpyAsync(m.s.pos_ctx, &p32, sizeof(p32),
                                   cudaMemcpyHostToDevice, stream),
                   "E5A3 commit absolute position");
        for (int layer = 0; layer < kLayers; ++layer) {
            const auto& lay = m.layers[layer];
            m.ws_kv->forward(lay.k, lay.km, m.s.hctx, m.s.kctx, 1, stream);
            m.ws_kv->forward(lay.v, lay.vm, m.s.hctx, m.s.vctx, 1, stream);
            const int slot = static_cast<int>(pos & kRingMask);
            if(m.gaming[Gopt::DraftRingNormRope]) {
                gopt_draft_ring_norm_rope_kernel<<<kKVHeads,128,128*sizeof(float),stream>>>(
                    m.s.kctx,m.s.vctx,lay.k_norm,m.s.pos_ctx,
                    m.ring_k[layer],m.ring_v[layer],kKVHeads);
                cuda_check(cudaGetLastError(),"GOPT commit ring norm/RoPE");
                gopt_record(m.gaming_submissions,Gopt::DraftRingNormRope);
            } else {
                dflash_head_norm_kernel<<<kKVHeads, 128, 128 * sizeof(float), stream>>>(
                    m.s.kctx, lay.k_norm, m.s.kctx_n, kKVHeads, kHeadDim);
                dflash_rope_kernel<<<(kKVHeads * kHeadDim + 255) / 256, 256, 0, stream>>>(
                    m.s.kctx_n, m.s.kctx_r, m.s.pos_ctx, 1, kKVHeads, kHeadDim);
                cuda_check(cudaGetLastError(), "E5A3 commit rope");
                dflash_ring_scatter_kernel<<<(kKVHeads * kHeadDim + 255) / 256, 256, 0, stream>>>(
                    m.s.kctx_r, m.s.vctx, m.ring_k[layer], m.ring_v[layer], slot);
                cuda_check(cudaGetLastError(), "E5A3 commit ring scatter");
            }
        }
        if (m.ring_count < kRingKeep) {
            ++m.ring_count;
        } else {
            ++m.ring_base_abs;  // ring full: evict oldest (reference trim)
        }
    }
}

void Exl3Dflash2DraftModel::commit_prefill_block_internal(const std::uint16_t* const* taps,
    int rows, long long abs_pos0, cudaStream_t stream) {
    impl_->invalidate_ring_witness();
    Impl& m = *impl_;
    require(taps && rows >= 1 && rows <= 16, "draft prefill commit requires1..16 rows");
    for (int t = 0; t < kLayers; ++t) require(taps[t] != nullptr, "draft prefill tap pointer missing");
    require(abs_pos0 >= 0 && abs_pos0 <= 2147483647LL - rows, "draft prefill position out of range");
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cuda_check(cudaStreamIsCapturing(stream, &capture), "draft prefill capture status");
    require(capture == cudaStreamCaptureStatusNone, "draft prefill commit requires eager stream");
    if (!m.prefill_batch || rows == 1) { commit_target_block_internal(taps, rows, abs_pos0, stream); return; }
    require(m.ring_count == 0 || abs_pos0 == m.ring_base_abs + m.ring_count,
            "draft prefill commit must extend committed span");
    const auto admission = Exl3CudaLinearAdmission::draft_prefill_fc;
    require(m.ws_fc->draft_prefill_fc_candidate(m.fc_meta, std::min(rows,8), admission),
            "draft prefill FC admission unavailable");
    for (const auto& layer : m.layers)
        require(m.ws_kv->draft_prefill_kv_candidate(layer.km, std::min(rows,8)) &&
                m.ws_kv->draft_prefill_kv_candidate(layer.vm, std::min(rows,8)),
                "draft prefill K/V admission unavailable");
    if (m.ring_count == 0) m.ring_base_abs = abs_pos0;
    for (int first = 0; first < rows; first += 8) {
        const int count = std::min(8, rows - first);
        const int pos = static_cast<int>(abs_pos0 + first);
        const std::uint16_t* ptrs[kLayers];
        for (int t = 0; t < kLayers; ++t) ptrs[t] = taps[t] + static_cast<std::size_t>(first) * kHidden;
        cuda_check(cudaMemcpyAsync(m.s.tap_ptr, ptrs, sizeof(ptrs), cudaMemcpyHostToDevice, stream),
                   "draft prefill tap pointers");
        dflash_tap_concat_kernel<<<(count * kLayers * kHidden + 255) / 256,256,0,stream>>>(m.s.tap_ptr,m.s.tcat,count);
        cuda_check(cudaGetLastError(), "draft prefill concat");
        m.ws_fc->forward(m.fc,m.fc_meta,m.s.tcat,m.s.fc_raw,count,stream,admission);
        dflash_rms_norm_kernel<DFlashFmt::F16><<<count,256,256*sizeof(float),stream>>>(
            m.s.fc_raw,m.hidden_norm,m.s.hctx,count,kHidden);
        cuda_check(cudaGetLastError(), "draft prefill hidden norm");
        dflash_prefill_positions_kernel<<<1,8,0,stream>>>(m.s.pos_ctx,count,pos);
        cuda_check(cudaGetLastError(), "draft prefill positions");
        for (int layer = 0; layer < kLayers; ++layer) {
            const auto& lay = m.layers[layer];
            m.ws_kv->forward(lay.k,lay.km,m.s.hctx,m.s.kctx,count,stream);
            m.ws_kv->forward(lay.v,lay.vm,m.s.hctx,m.s.vctx,count,stream);
            if(m.gaming[Gopt::DraftRingNormRope]) {
                gopt_draft_ring_norm_rope_kernel<<<count*kKVHeads,128,128*sizeof(float),stream>>>(
                    m.s.kctx,m.s.vctx,lay.k_norm,m.s.pos_ctx,
                    m.ring_k[layer],m.ring_v[layer],count*kKVHeads);
                cuda_check(cudaGetLastError(),"GOPT prefill ring norm/RoPE");
                gopt_record(m.gaming_submissions,Gopt::DraftRingNormRope);
            } else {
                dflash_head_norm_kernel<<<count*kKVHeads,128,128*sizeof(float),stream>>>(
                    m.s.kctx,lay.k_norm,m.s.kctx_n,count*kKVHeads,kHeadDim);
                dflash_rope_kernel<<<(count*kKVHeads*kHeadDim+255)/256,256,0,stream>>>(
                    m.s.kctx_n,m.s.kctx_r,m.s.pos_ctx,count,kKVHeads,kHeadDim);
                cuda_check(cudaGetLastError(), "draft prefill rope");
                dflash_prefill_ring_scatter_kernel<<<(count*kKVHeads*kHeadDim+255)/256,256,0,stream>>>(
                    m.s.kctx_r,m.s.vctx,m.ring_k[layer],m.ring_v[layer],count,pos);
                cuda_check(cudaGetLastError(), "draft prefill ring scatter");
            }
        }
        const int dropped = std::max(0,m.ring_count+count-kRingKeep);
        m.ring_count = std::min(kRingKeep,m.ring_count+count);
        m.ring_base_abs += dropped;
    }
}

std::unique_ptr<Exl3Dflash2DraftModel> Exl3Dflash2DraftModel::load(
    const std::filesystem::path& model_directory) {
    return load_impl(model_directory,false);
}
std::unique_ptr<Exl3Dflash2DraftModel> Exl3Dflash2DraftModel::load_impl(
    const std::filesystem::path& model_directory,bool defer_private) {
    auto impl = std::make_unique<Impl>();
    cuda_check(cudaGetDevice(&impl->device),"draft weight device");
    const char* prefill_batch = std::getenv("NINFER_DFLASH2_PREFILL_BATCH");
    impl->prefill_batch = prefill_batch && std::string(prefill_batch) == "1";
    const char* target_commit_batch = std::getenv("NINFER_DFLASH2_TARGET_COMMIT_BATCH");
    require(target_commit_batch == nullptr || std::string(target_commit_batch) == "0" ||
                std::string(target_commit_batch) == "1",
            "NINFER_DFLASH2_TARGET_COMMIT_BATCH must be 0 or 1");
    impl->target_commit_batch = target_commit_batch != nullptr &&
        std::string(target_commit_batch) == "1";
    require(!impl->target_commit_batch || impl->prefill_batch,
            "NINFER_DFLASH2_TARGET_COMMIT_BATCH requires NINFER_DFLASH2_PREFILL_BATCH=1");
    const char* parallel_topk = std::getenv("NINFER_DFLASH2_PARALLEL_TOPK");
    impl->parallel_topk = parallel_topk == nullptr || std::string(parallel_topk) != "0";
    const char* local_merge_topk = std::getenv("NINFER_DFLASH2_LOCAL_TOPK");
    impl->local_merge_topk = local_merge_topk && std::string(local_merge_topk) == "1";
    const char* parallel_ring = std::getenv("NINFER_DFLASH2_PARALLEL_RING_ATTENTION");
    impl->parallel_ring_attention = parallel_ring == nullptr || std::string(parallel_ring) == "1";
    const char* position_confidence =
        std::getenv("NINFER_DFLASH2_POSITION_CONFIDENCE");
    require(position_confidence == nullptr ||
                std::string(position_confidence) == "0" ||
                std::string(position_confidence) == "1",
            "NINFER_DFLASH2_POSITION_CONFIDENCE must be 0 or 1");
    impl->position_confidence =
        position_confidence != nullptr && std::string(position_confidence) == "1";
    const char* dense_kmajor=std::getenv("NINFER_DFLASH2_DENSE_KMAJOR");
    require(dense_kmajor==nullptr || std::string(dense_kmajor)=="0" ||
            std::string(dense_kmajor)=="1",
        "NINFER_DFLASH2_DENSE_KMAJOR must be 0 or 1");
    impl->dense_kmajor=dense_kmajor!=nullptr && std::string(dense_kmajor)=="1";
    const char* selector_batched_anchor_chain =
        std::getenv("NINFER_DFLASH2_FAST_SELECTOR_BATCHED_ANCHOR_CHAIN");
    require(selector_batched_anchor_chain == nullptr ||
                std::string(selector_batched_anchor_chain) == "0" ||
                std::string(selector_batched_anchor_chain) == "1",
            "NINFER_DFLASH2_FAST_SELECTOR_BATCHED_ANCHOR_CHAIN must be 0 or 1");
    impl->selector_batched_anchor_chain =
        selector_batched_anchor_chain != nullptr &&
        std::string(selector_batched_anchor_chain) == "1";
    const char* fast_device_liveness=std::getenv("NINFER_DFLASH2_FAST_DEVICE_LIVENESS");
    require(fast_device_liveness==nullptr || std::string(fast_device_liveness)=="0" ||
            std::string(fast_device_liveness)=="1",
        "NINFER_DFLASH2_FAST_DEVICE_LIVENESS must be 0 or 1");
    impl->fast_device_liveness=fast_device_liveness!=nullptr &&
        std::string(fast_device_liveness)=="1";
    const char* fused_topk_liveness=std::getenv("NINFER_DFLASH2_FUSED_TOPK_LIVENESS");
    require(fused_topk_liveness==nullptr || std::string(fused_topk_liveness)=="0" ||
            std::string(fused_topk_liveness)=="1",
        "NINFER_DFLASH2_FUSED_TOPK_LIVENESS must be 0 or 1");
    impl->fused_topk_liveness=fused_topk_liveness!=nullptr &&
        std::string(fused_topk_liveness)=="1";
    require(!impl->fused_topk_liveness ||
                (impl->fast_device_liveness && impl->local_merge_topk),
        "fused top-K liveness requires fast device liveness and local top-K");
    const char* fused_selector=std::getenv("NINFER_DFLASH2_FUSED_SELECTOR");
    require(fused_selector==nullptr || std::string(fused_selector)=="0" ||
            std::string(fused_selector)=="1",
        "NINFER_DFLASH2_FUSED_SELECTOR must be 0 or 1");
    impl->fused_selector=fused_selector!=nullptr &&
        std::string(fused_selector)=="1";
    // Capture the private allocation requirement before timing resources or weight
    // uploads. Construction must still agree with this configuration-derived plan.
    impl->required_execution_bytes=impl->plan_private_execution().total_bytes;
    if(!defer_private)impl->projection_timing.initialize();
    impl->directory = model_directory;

    const auto config_path = model_directory / "config.json";
    std::ifstream config_file(config_path, std::ios::binary);
    require(static_cast<bool>(config_file), "cannot open DFlash2 config.json");
    Json config = Json::parse(
        std::string(std::istreambuf_iterator<char>(config_file), {}));
    require(config.at("architectures").at(0).get<std::string>() == "DFlash2DraftModel",
            "E5A2 artifact is not DFlash2DraftModel");
    const auto& df = config.at("dflash_config");
    require(df.at("block_size").get<int>() == kBlockCap &&
            df.at("conv_group_size").get<int>() == kConvGroup &&
            df.at("conv_kernel_size").get<int>() == kConvKernel &&
            df.at("mask_token_id").get<int>() == 248070 &&
            df.at("selector_rank").get<int>() == kRank &&
            df.at("selector_top_k").get<int>() == kTopK,
            "E5A2 dflash_config mismatch");
    const auto& tap_ids = df.at("target_layer_ids");
    require(tap_ids.size() == kLayers, "E5A2 target_layer_ids length mismatch");
    for (int i = 0; i < kLayers; ++i) {
        require(tap_ids[i].get<int>() == kTapLayers[i],
                "E5A2 target_layer_ids order mismatch");
    }
    require(config.at("hidden_size").get<int>() == kHidden &&
            config.at("num_hidden_layers").get<int>() == kLayers &&
            config.at("intermediate_size").get<int>() == kIntermediate &&
            config.at("vocab_size").get<int>() == kVocab &&
            config.at("num_attention_heads").get<int>() == kQHeads &&
            config.at("num_key_value_heads").get<int>() == kKVHeads &&
            config.at("head_dim").get<int>() == kHeadDim &&
            config.at("is_causal").get<bool>() == false &&
            config.at("sliding_window").get<int>() == 2048,
            "E5A2 draft geometry mismatch");
    const auto& q = config.at("quantization_config");
    require(q.at("quant_method").get<std::string>() == "exl3" &&
            q.at("version").get<std::string>() == "1.4.2" &&
            q.at("codebook").get<std::string>() == "mul1" &&
            q.at("out_scales").get<std::string>() == "always",
            "E5A2 unsupported DFlash2 quantization metadata");

    const auto safetensors_path = model_directory / "model.safetensors";
    impl->header = inspect_file(safetensors_path);
    require(impl->header.tensors.size() == 189,
            "E5A2 DFlash2 safetensors tensor count mismatch");

    auto& allocs = *impl->allocations;
    auto& total = impl->weight_bytes;
    std::unordered_set<std::string> requested;

    auto load_linear = [&](const std::string& prefix, int in, int out,
                           Exl3CudaLinearMetadata& md,
                           const std::string& label) -> Exl3CudaLinearWeights {
        const auto trellis = load_tensor_from_file(safetensors_path, impl->header,
                                                   prefix + ".trellis");
        const auto suh = load_tensor_from_file(safetensors_path, impl->header,
                                               prefix + ".suh");
        const auto svh = load_tensor_from_file(safetensors_path, impl->header,
                                               prefix + ".svh");
        const auto mul1 = load_tensor_from_file(safetensors_path, impl->header,
                                                prefix + ".mul1");
        require(trellis.info.dtype == "I16" && trellis.info.shape.size() == 3 &&
                trellis.info.shape[0] == static_cast<std::uint64_t>(in / 16) &&
                trellis.info.shape[1] == static_cast<std::uint64_t>(out / 16) &&
                trellis.info.shape[2] == 80,
                prefix + " EXL3 trellis mismatch");
        require(suh.info.dtype == "F16" &&
                suh.info.shape ==
                    std::vector<std::uint64_t>{static_cast<std::uint64_t>(in)},
                prefix + " suh mismatch");
        require(svh.info.dtype == "F16" &&
                svh.info.shape ==
                    std::vector<std::uint64_t>{static_cast<std::uint64_t>(out)},
                prefix + " svh mismatch");
        require(mul1.info.dtype == "I32" && mul1.info.shape.empty(),
                prefix + " mul1 mismatch");
        requested.insert(prefix + ".trellis");
        requested.insert(prefix + ".suh");
        requested.insert(prefix + ".svh");
        requested.insert(prefix + ".mul1");
        Exl3CudaLinearMetadata result{in, out, 5, false, true, false};
        md = result;
        Exl3CudaLinearWeights weights;
        weights.trellis = reinterpret_cast<const std::uint16_t*>(
            upload_raw16(trellis, label.c_str(), "I16", allocs, total));
        weights.suh = upload_raw16(suh, label.c_str(), "F16", allocs, total);
        weights.svh = upload_raw16(svh, label.c_str(), "F16", allocs, total);
        weights.mul1 = reinterpret_cast<const std::int32_t*>(
            upload_raw16(mul1, label.c_str(), "I32", allocs, total));
        return weights;
    };

    auto load_norm = [&](const std::string& name, int features)
        -> const std::uint16_t* {
        const auto tensor = load_tensor_from_file(safetensors_path, impl->header, name);
        require(tensor.info.shape ==
                    std::vector<std::uint64_t>{static_cast<std::uint64_t>(features)},
                name + " shape mismatch");
        requested.insert(name);
        return upload_f16_or_bf16(tensor, name.c_str(), allocs, total);
    };

    auto load_conv = [&](const std::string& conv, const std::uint16_t** base_out,
                         const std::uint16_t** proj_out) {
        const auto base_kernel = load_tensor_from_file(
            safetensors_path, impl->header, conv + ".base_kernel");
        require(base_kernel.info.dtype == "BF16" &&
                base_kernel.info.shape == std::vector<std::uint64_t>{
                                              2, 2, static_cast<std::uint64_t>(kHidden)},
                conv + " base_kernel mismatch");
        const auto kernel_projection = load_tensor_from_file(
            safetensors_path, impl->header, conv + ".kernel_projection.weight");
        require(kernel_projection.info.dtype == "F16" &&
                kernel_projection.info.shape == std::vector<std::uint64_t>{
                                                    static_cast<std::uint64_t>(kConvDynamic),
                                                    static_cast<std::uint64_t>(kHidden)},
                conv + " kernel_projection mismatch");
        requested.insert(conv + ".base_kernel");
        requested.insert(conv + ".kernel_projection.weight");
        *base_out = upload_f16_or_bf16(base_kernel, (conv + " base").c_str(), allocs, total);
        *proj_out = impl->dense_kmajor?
            upload_kmajor_f16(kernel_projection,kConvDynamic,kHidden,
                (conv+" proj K-major").c_str(),allocs,total):
            upload_raw16(kernel_projection,(conv+" proj").c_str(),"F16",allocs,total);
    };

    impl->fc = load_linear("fc", kLayers * kHidden, kHidden, impl->fc_meta,
                           "E5A2 draft fc");
    impl->hidden_norm = load_norm("hidden_norm.weight", kHidden);
    impl->norm = load_norm("norm.weight", kHidden);

    for (int i = 0; i < kLayers; ++i) {
        auto& layer = impl->layers[i];
        const std::string base = "layers." + std::to_string(i);
        layer.q = load_linear(base + ".self_attn.q_proj", kHidden,
                              kQHeads * kHeadDim, layer.qm, base + " q_proj");
        layer.k = load_linear(base + ".self_attn.k_proj", kHidden,
                              kKVHeads * kHeadDim, layer.km, base + " k_proj");
        layer.v = load_linear(base + ".self_attn.v_proj", kHidden,
                              kKVHeads * kHeadDim, layer.vm, base + " v_proj");
        layer.o = load_linear(base + ".self_attn.o_proj", kQHeads * kHeadDim,
                              kHidden, layer.om, base + " o_proj");
        layer.gate = load_linear(base + ".mlp.gate_proj", kHidden, kIntermediate,
                                 layer.gm, base + " gate");
        layer.up = load_linear(base + ".mlp.up_proj", kHidden, kIntermediate,
                               layer.um, base + " up");
        layer.down = load_linear(base + ".mlp.down_proj", kIntermediate, kHidden,
                                 layer.dm, base + " down");
        layer.input_norm = load_norm(base + ".input_layernorm.weight", kHidden);
        layer.post_norm = load_norm(base + ".post_attention_layernorm.weight", kHidden);
        layer.q_norm = load_norm(base + ".self_attn.q_norm.weight", kHeadDim);
        layer.k_norm = load_norm(base + ".self_attn.k_norm.weight", kHeadDim);
        load_conv(base + ".attention_conv", &layer.attn_base, &layer.attn_kernel_proj);
        load_conv(base + ".mlp_conv", &layer.mlp_base, &layer.mlp_kernel_proj);
    }

    const auto hidden_projection = load_tensor_from_file(
        safetensors_path, impl->header, "candidate_selector.hidden_projection.weight");
    require(hidden_projection.info.dtype == "F16" &&
            hidden_projection.info.shape == std::vector<std::uint64_t>{
                                                static_cast<std::uint64_t>(kRank),
                                                static_cast<std::uint64_t>(kHidden)},
            "candidate_selector.hidden_projection.weight mismatch");
    requested.insert("candidate_selector.hidden_projection.weight");
    impl->hidden_proj = impl->dense_kmajor?
        upload_kmajor_f16(hidden_projection,kRank,kHidden,
            "E5A2 candidate hidden_projection K-major",allocs,total):
        upload_raw16(hidden_projection,
            "E5A2 candidate hidden_projection","F16",allocs,total);

    auto load_codebook = [&](const std::string& name) -> const std::uint16_t* {
        const auto tensor = load_tensor_from_file(safetensors_path, impl->header, name);
        require(tensor.info.dtype == "BF16" &&
                tensor.info.shape == std::vector<std::uint64_t>{
                                        static_cast<std::uint64_t>(kVocab),
                                        static_cast<std::uint64_t>(kRank)},
                name + " codebook mismatch");
        requested.insert(name);
        return upload_f16_or_bf16(tensor, name.c_str(), allocs, total);
    };
    impl->pred_cb = load_codebook("candidate_selector.predecessor_codebook");
    impl->succ_cb = load_codebook("candidate_selector.successor_codebook");

    for (const auto& info : impl->header.tensors) {
        require(requested.count(info.name) == 1,
                "E5A2 unexpected tensor in DFlash2 artifact: " + info.name);
    }

    if(!defer_private)impl->allocate_execution(impl->required_execution_bytes);

    return std::unique_ptr<Exl3Dflash2DraftModel>(
        new Exl3Dflash2DraftModel(std::move(impl)));
}
std::vector<std::int64_t> Exl3Dflash2DraftModel::propose(
    const std::vector<std::int64_t>& block_ids,
    int block_pos0,
    const std::uint16_t* const* layer_taps,
    int context_rows,
    int ctx_pos0,
    int window_rows,
    const std::uint16_t* target_embedding_bf16,
    const Exl3CudaLinearWeights& target_head,
    const Exl3CudaLinearMetadata& target_head_metadata,
    std::int64_t mask_token_id,
    cudaStream_t stream) {
    // E5A2 qualified path: recompute the window from taps every cycle.
    return propose_internal(block_ids, block_pos0, layer_taps, context_rows,
                            ctx_pos0, window_rows, target_embedding_bf16,
                            target_head, target_head_metadata, mask_token_id,
                            stream, false);
}

std::vector<std::int64_t> Exl3Dflash2DraftModel::propose_cached(
    const std::vector<std::int64_t>& block_ids, int block_pos0,
    const std::uint16_t* target_embedding_bf16,
    const Exl3CudaLinearWeights& target_head,
    const Exl3CudaLinearMetadata& target_head_metadata,
    std::int64_t mask_token_id, cudaStream_t stream) {
    // E5A3 ring path: committed ring K/V replace the recomputed window.
    return propose_internal(block_ids, block_pos0, nullptr, 0, 0, 0,
                            target_embedding_bf16, target_head,
                            target_head_metadata, mask_token_id, stream, true);
}

std::vector<std::int64_t> Exl3Dflash2DraftModel::propose_cached_view(
    std::span<const std::int64_t> block_ids,int block_pos0,
    const std::uint16_t* target_embedding_bf16,const Exl3CudaLinearWeights& target_head,
    const Exl3CudaLinearMetadata& target_head_metadata,std::int64_t mask_token_id,cudaStream_t stream) {
    return propose_internal(block_ids,block_pos0,nullptr,0,0,0,target_embedding_bf16,
        target_head,target_head_metadata,mask_token_id,stream,true);
}

std::vector<std::int64_t> Exl3Dflash2DraftModel::propose_cached_device_seed(
    std::span<const std::int64_t> masked_block,const Exl3DeviceGreedySeed& seed,
    int block_pos0,const std::uint16_t* target_embedding_bf16,
    const Exl3CudaLinearWeights& target_head,
    const Exl3CudaLinearMetadata& target_head_metadata,
    std::int64_t mask_token_id,cudaStream_t stream) {
    return propose_internal(masked_block,block_pos0,nullptr,0,0,0,
        target_embedding_bf16,target_head,target_head_metadata,mask_token_id,
        stream,true,&seed);
}

std::vector<std::int64_t> Exl3Dflash2DraftModel::propose_internal(
    std::span<const std::int64_t> block_ids,
    int block_pos0,
    const std::uint16_t* const* layer_taps,
    int context_rows,
    int ctx_pos0,
    int window_rows,
    const std::uint16_t* target_embedding_bf16,
    const Exl3CudaLinearWeights& target_head,
    const Exl3CudaLinearMetadata& target_head_metadata,
    std::int64_t mask_token_id,
    cudaStream_t stream, bool use_ring,const Exl3DeviceGreedySeed* device_seed) {
    require_no_fresh_prefill();
    Impl& m = *impl_;
    struct HostControlScope {
        bool& active;
        explicit HostControlScope(bool& value):active(value) {
            require(!active,"reentrant draft host control mutation");active=true;
        }
        ~HostControlScope(){active=false;}
    } host_control_scope(m.host_control_active);
    require(m.host_control_generation!=std::numeric_limits<std::uint64_t>::max(),
        "draft host control generation exhausted");
    ++m.host_control_generation;
    const auto seed_ring_parent=device_seed?m.host_ring_parent:nullptr;
    const auto seed_ring_revision=m.ring_revision;
    require(block_ids.size()>=2 && block_ids.size()<=static_cast<std::size_t>(m.block_capacity),
            "E5A2 block length outside E5A2 capacity");
    const int block_len = static_cast<int>(block_ids.size());
    require(block_pos0>=0 && block_pos0<=std::numeric_limits<int>::max()-block_len,
        "E5A2 draft block position overflow");
    require(std::all_of(block_ids.begin(),block_ids.end(),[](std::int64_t token){return token>=0 && token<kVocab;}),
        "E5A2 draft block token outside vocabulary");
    if(device_seed) {
        const auto& seed_head=device_seed->target_head;
        const auto& seed_meta=device_seed->target_head_metadata;
        require(use_ring && device_seed->owner && device_seed->model_owner &&
                    device_seed->model_identity && device_seed->device_row &&
                    device_seed->target_embedding_bf16==target_embedding_bf16 &&
                    seed_head.trellis==target_head.trellis &&
                    seed_head.suh==target_head.suh && seed_head.svh==target_head.svh &&
                    seed_head.mul1==target_head.mul1 &&
                    seed_meta.in_features==target_head_metadata.in_features &&
                    seed_meta.out_features==target_head_metadata.out_features &&
                    seed_meta.K==target_head_metadata.K && seed_meta.mcg==target_head_metadata.mcg &&
                    seed_meta.mul1==target_head_metadata.mul1 &&
                    seed_meta.has_bias==target_head_metadata.has_bias &&
                    device_seed->producer_ready && device_seed->consumer_done &&
                    device_seed->consumer_claimed && device_seed->consumer_recorded &&
                    device_seed->consumer_stream && device_seed->acquisition &&
                    device_seed->execution && device_seed->generation &&
                    device_seed->serial && device_seed->position==block_pos0 &&
                    device_seed->acquisition==m.ring_acquisition &&
                    device_seed->execution==m.ring_execution &&
                    host_ring_lineage(seed_ring_parent,m.ring_acquisition,m.ring_execution) &&
                    std::all_of(block_ids.begin(),block_ids.end(),
                        [=](std::int64_t token){return token==mask_token_id;}),
                "E5A3 device seed owner/scope/mask mismatch");
    }
    require(target_embedding_bf16 != nullptr,
            "E5A2 draft propose needs target embeddings");
    const int proposal_rows = block_len - 1;  // E5A2 logits_start == 1
    int ctx_skip = 0;
    int ctx_begin_pos = 0;
    int eff_window = 0;
    if (use_ring) {
        require(m.ring_count >= 0 && m.ring_count <= kRingKeep,
                "E5A3 ring length outside bounded capacity");
        require(m.ring_base_abs>=0 && m.ring_base_abs<=std::numeric_limits<int>::max()-m.ring_count,
                "E5A3 committed ring position overflow");
        require(static_cast<std::int64_t>(block_pos0) == m.ring_base_abs + m.ring_count,
                "E5A3 cached block must immediately extend the committed span");
        eff_window = m.ring_count;
    } else {
        require(window_rows >= 1 && window_rows <= context_rows &&
                window_rows <= m.context_capacity,
                "E5A2 draft context window outside E5A2 capacity");
        require(layer_taps != nullptr,
                "E5A2 draft propose needs target taps");
        require(ctx_pos0>=0 && context_rows<=std::numeric_limits<int>::max()-ctx_pos0,
                "E5A2 draft context position overflow");
        ctx_skip = context_rows - window_rows;
        ctx_begin_pos = ctx_pos0 + ctx_skip;
        require(block_pos0 == ctx_begin_pos + window_rows,
                "E5A2 draft block must immediately follow the context window");
        eff_window = window_rows;
    }
    (void)eff_window;
    if(device_seed) {
        require(m.ring_revision==seed_ring_revision &&
                    host_ring_lineage(seed_ring_parent,device_seed->acquisition,
                        device_seed->execution),
                "E5A3 device seed ring witness changed before claim");
        bool unclaimed=false;
        require(device_seed->consumer_claimed->compare_exchange_strong(
                    unclaimed,true,std::memory_order_acq_rel),
                "E5A3 device seed already consumed");
        *device_seed->consumer_stream=stream;
    }

    auto& projection_timing = m.projection_timing;
    std::chrono::steady_clock::time_point projection_api_begin{};
    require(m.projection_observer == nullptr || !projection_timing.enabled,
            "draft projection observation is incompatible with projection timing");
    require(m.ring_attention_observer == nullptr || !projection_timing.enabled,
            "ring attention observation is incompatible with projection timing");
    if (m.ring_attention_observer != nullptr) {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        cuda_check(cudaStreamIsCapturing(stream, &status), "ring observation capture query");
        require(status == cudaStreamCaptureStatusNone,
                "ring attention observation supports eager propose only");
    }
    if (m.projection_observer != nullptr) {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        cuda_check(cudaStreamIsCapturing(stream, &capture_status),
                   "draft projection observation capture query");
        require(capture_status == cudaStreamCaptureStatusNone,
                "draft projection observation supports eager propose only");
    }
    if (projection_timing.enabled) {
        for (const char* incompatible : {
                 "NINFER_DFLASH2_TIMING", "NINFER_DFLASH2_DEBUG_LOGITS",
                 "NINFER_DFLASH2_REF_FP32", "NINFER_DFLASH2_DIFF_L0_OPROJ",
                 "NINFER_DFLASH2_DIFF_L0_ATTN", "NINFER_DFLASH2_DIFF_L0_CONV"}) {
            require(std::getenv(incompatible) == nullptr,
                    std::string("NINFER_DFLASH2_PROJECTION_TIMING is incompatible with ") +
                        incompatible);
        }
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        cuda_check(cudaStreamIsCapturing(stream, &capture_status),
                   "P2 projection timing capture query");
        require(capture_status == cudaStreamCaptureStatusNone,
                "NINFER_DFLASH2_PROJECTION_TIMING supports eager propose only");
        projection_api_begin = std::chrono::steady_clock::now();
        projection_timing.begin(stream);
    }
    auto timing_close = [&](Impl::TimingCategory category, const char* scope) {
        if (!projection_timing.enabled) return;
        projection_timing.close(category, scope, -1, 0, nullptr, nullptr, stream);
    };
    // Hold the completed immutable conditioning parent through every suspension.
    // A geometric ring match alone does not establish completed tap provenance.
    const auto shared_ring_parent=m.host_ring_parent;
    Exl3ActivationLifetime::Witness shared_mlp_activation{};
    const bool shared_segment_enabled=m.shared_q_executor && use_ring && block_len==8 &&
        !projection_timing.enabled && !m.projection_observer &&
        host_ring_lineage(shared_ring_parent,m.ring_acquisition,m.ring_execution);
    Exl3DraftPrivateSegment shared_segment;
    const auto shared_ring_revision=m.ring_revision;
    if(shared_segment_enabled)shared_segment=Exl3DraftPrivateSegment::make(
        m.ring_acquisition,m.ring_execution,m.ring_base_abs,m.ring_count,block_pos0,block_ids);
    auto timed_forward = [&](Exl3CudaLinearWorkspace& workspace,
                             const Exl3CudaLinearWeights& weights,
                             const Exl3CudaLinearMetadata& metadata,
                             const std::uint16_t* input, std::uint16_t* output,
                             int rows, int layer, const char* scope,
                             Impl::TimingCategory category) {
        const char* dispatch = projection_timing.enabled
            ? workspace.dispatch_name(metadata, rows) : nullptr;
        const bool observed_k5 = m.projection_observer != nullptr && layer >= 0 &&
            category == Impl::TimingCategory::Projection &&
            (std::strcmp(scope, "q") == 0 || std::strcmp(scope, "k") == 0 ||
             std::strcmp(scope, "v") == 0 || std::strcmp(scope, "o") == 0 ||
             std::strcmp(scope, "down") == 0 || (m.projection_observer_gateup &&
             (std::strcmp(scope,"gate")==0 || std::strcmp(scope,"up")==0)));
        if (observed_k5) {
            ProjectionObservation observation{
                weights, metadata, input, rows, layer, scope, stream};
            m.projection_observer(observation, m.projection_observer_user);
        }
        const bool block_k=m.shared_block_kv && std::strcmp(scope,"k")==0;
        const bool block_v=m.shared_block_kv && std::strcmp(scope,"v")==0;
        const bool block_o=m.shared_block_o && std::strcmp(scope,"o")==0;
        const bool block_down=m.shared_block_down && std::strcmp(scope,"down")==0;
        const bool block_gate=m.shared_block_gateup && std::strcmp(scope,"gate")==0;
        const bool block_up=m.shared_block_gateup && std::strcmp(scope,"up")==0;
        bool shared_q=false;
        const auto shared_family=block_gate?Exl3TargetSharedFamily::draft_gate:
            block_up?Exl3TargetSharedFamily::draft_up:block_down?Exl3TargetSharedFamily::draft_down:
            block_o?Exl3TargetSharedFamily::draft_o:block_k?Exl3TargetSharedFamily::draft_k:
            block_v?Exl3TargetSharedFamily::draft_v:Exl3TargetSharedFamily::draft_q;
        if(shared_segment_enabled && rows==8 &&
            layer>=0 && (std::strcmp(scope,"q")==0 || block_k || block_v || block_o || block_down || block_gate || block_up) &&
            target_shared_admission(shared_family,metadata).has_value()) {
            const auto require_segment=[&] {
                require(m.ring_revision==shared_ring_revision &&
                    host_ring_lineage(shared_ring_parent,m.ring_acquisition,m.ring_execution) &&
                    shared_segment.matches_tokens(block_ids) &&
                    shared_segment.matches(m.ring_acquisition,m.ring_execution,m.ring_base_abs,
                    m.ring_count,block_pos0,block_ids[0]),"shared draft proposal conditioning changed");
            };
            require_segment();
            shared_q=m.shared_q_executor(Exl3DraftSharedQContinuation{
                {weights,metadata,input,output,rows,block_pos0,layer,stream,
                    shared_family,target_head.trellis,(block_gate || block_up)?shared_mlp_activation:Exl3ActivationLifetime::Witness{},
                    {target_head.trellis,target_head.suh,target_head.svh,target_head.mul1},target_head_metadata},
                m.ring_acquisition,m.ring_execution,block_ids[0],m.ring_base_abs,m.ring_count,
                shared_segment,shared_ring_parent});
            require_segment(); // including a callback that requests serial fallback
        }
        if(!shared_q)workspace.forward(weights, metadata, input, output, rows, stream);
        if (projection_timing.enabled) {
            projection_timing.close(category, scope, layer, rows, &metadata,
                                    dispatch, stream);
        }
    };

    // E5A3 latency attribution (env-gated NINFER_DFLASH2_TIMING; serializes phases
    // when active, so GPU times are exact but the timed run is not a perf baseline).
    const bool e5a3_timing = (std::getenv("NINFER_DFLASH2_TIMING") != nullptr);
    cudaEvent_t ph_a = nullptr, ph_b = nullptr;
    std::vector<std::pair<std::string, float>> ph_ms;
    auto ph_begin = [&]() {
        if (!e5a3_timing) return;
        cuda_check(cudaEventCreate(&ph_a), "E5A3 timing create a");
        cuda_check(cudaEventCreate(&ph_b), "E5A3 timing create b");
        cuda_check(cudaEventRecord(ph_a, stream), "E5A3 timing begin");
    };
    auto ph_end = [&](const std::string& name) {
        if (!e5a3_timing) return;
        cuda_check(cudaEventRecord(ph_b, stream), "E5A3 timing end");
        cuda_check(cudaEventSynchronize(ph_b), "E5A3 timing sync");
        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, ph_a, ph_b), "E5A3 timing elapsed");
        ph_ms.emplace_back(name, ms);
        cuda_check(cudaEventDestroy(ph_a), "E5A3 timing destroy a");
        cuda_check(cudaEventDestroy(ph_b), "E5A3 timing destroy b");
    };
    cudaEvent_t tot_a = nullptr, tot_b = nullptr;
    if (e5a3_timing) {
        cuda_check(cudaEventCreate(&tot_a), "E5A3 timing total a");
        cuda_check(cudaEventCreate(&tot_b), "E5A3 timing total b");
        cuda_check(cudaEventRecord(tot_a, stream), "E5A3 timing total begin");
    }
    ph_begin();  // uploads phase

    std::array<const std::uint16_t*, kLayers> tap_base{};
    if (!use_ring) {
        for (int t = 0; t < kLayers; ++t) {
            tap_base[t] = layer_taps[t] + static_cast<std::size_t>(ctx_skip) * kHidden;
        }
        cuda_check(cudaMemcpyAsync(m.s.tap_ptr, tap_base.data(),
                                   kLayers * sizeof(void*), cudaMemcpyHostToDevice, stream),
                   "E5A2 upload draft tap pointer array");
    }
    cuda_check(cudaMemcpyAsync(m.s.ids, block_ids.data(),
                               static_cast<std::size_t>(block_len) * sizeof(std::int64_t),
                               cudaMemcpyHostToDevice, stream),
               "E5A2 upload draft block ids");
    if(device_seed) {
        cuda_check(cudaStreamWaitEvent(stream,device_seed->producer_ready,0),
            "E5A3 wait authoritative device seed");
        dflash_prepare_device_seed_kernel<<<1,1,0,stream>>>(
            device_seed->device_row,device_seed->serial,mask_token_id,m.s.ids,
            m.s.proposal_out+7);
        cuda_check(cudaGetLastError(),"E5A3 prepare authoritative device seed");
        cuda_check(cudaEventRecord(device_seed->consumer_done,stream),
            "E5A3 record device seed final use");
        device_seed->consumer_recorded->store(true,std::memory_order_release);
    }
    require(block_len<=static_cast<int>(m.host_block_positions.size()) &&
                (use_ring || window_rows<=static_cast<int>(m.host_context_positions.size())),
        "draft host control extent exceeds fixed storage");
    auto host_pos_ctx=std::span<std::int32_t>(m.host_context_positions).first(
        use_ring?0:static_cast<std::size_t>(window_rows));
    auto host_pos_blk=std::span<std::int32_t>(m.host_block_positions).first(block_len);
    for (int i = 0; i < window_rows && !use_ring; ++i) host_pos_ctx[i] = ctx_begin_pos + i;
    for (int i = 0; i < block_len; ++i) host_pos_blk[i] = block_pos0 + i;
    if (!use_ring) {
        cuda_check(cudaMemcpyAsync(m.s.pos_ctx, host_pos_ctx.data(),
                                   host_pos_ctx.size() * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream),
                   "E5A2 upload draft ctx positions");
    }
    cuda_check(cudaMemcpyAsync(m.s.pos_blk, host_pos_blk.data(),
                               host_pos_blk.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream),
               "E5A2 upload draft block positions");
    ph_end("uploads");
    ph_begin();

    auto launch_rms = [&](const std::uint16_t* in, const std::uint16_t* weight,
                          std::uint16_t* out, int rows, int features,
                          bool in_bf16 = false) {
        if (in_bf16) {
            dflash_rms_norm_kernel<DFlashFmt::BF16><<<rows, 256, 256 * sizeof(float), stream>>>(
                in, weight, out, rows, features);
        } else {
            dflash_rms_norm_kernel<DFlashFmt::F16><<<rows, 256, 256 * sizeof(float), stream>>>(
                in, weight, out, rows, features);
        }
        cuda_check(cudaGetLastError(), "E5A2 launch draft RMSNorm");
    };
    auto launch_head_norm = [&](const std::uint16_t* in, const std::uint16_t* weight,
                                std::uint16_t* out, int segments, int head_dim) {
        dflash_head_norm_kernel<<<segments, 128, 128 * sizeof(float), stream>>>(
            in, weight, out, segments, head_dim);
        cuda_check(cudaGetLastError(), "E5A2 launch draft head norm");
    };
    auto launch_dense = [&](const std::uint16_t* in, const std::uint16_t* weight_outmajor,
                            std::uint16_t* out, int rows, int k, int n) {
        if(m.gaming[Gopt::DraftDenseRowPair] && rows>=2 && rows<=8 && k<=kHidden) {
            if(m.dense_kmajor)
                gopt_draft_dense_rowpair_kernel<true><<<dim3((n+255)/256,(rows+1)/2),256,
                    static_cast<std::size_t>(2*k)*sizeof(std::uint16_t),stream>>>(in,weight_outmajor,out,rows,k,n);
            else
                gopt_draft_dense_rowpair_kernel<false><<<dim3((n+255)/256,(rows+1)/2),256,
                    static_cast<std::size_t>(2*k)*sizeof(std::uint16_t),stream>>>(in,weight_outmajor,out,rows,k,n);
            cuda_check(cudaGetLastError(),"launch GOPT draft dense row pair");
            gopt_record(m.gaming_submissions,Gopt::DraftDenseRowPair);
        } else if(m.dense_kmajor &&
                  launch_dflash_dense_rows_kmajor(in,weight_outmajor,out,rows,k,n,stream)) {
            ++m.dense_kmajor_launches;
        } else if(m.dense_kmajor) {
            dflash_dense_gemm_t_kernel<true><<<dim3((n+255)/256,rows),256,
                static_cast<std::size_t>(k)*sizeof(std::uint16_t),stream>>>(
                in,weight_outmajor,out,rows,k,n);
            ++m.dense_kmajor_launches;
        } else {
            dflash_dense_gemm_t_kernel<false><<<dim3((n+255)/256,rows),256,
                static_cast<std::size_t>(k)*sizeof(std::uint16_t),stream>>>(
                in,weight_outmajor,out,rows,k,n);
        }
        cuda_check(cudaGetLastError(), "E5A2 launch draft dense GEMM");
    };
    auto launch_dyn_conv = [&](const std::uint16_t* x, const std::uint16_t* dyn,
                               const std::uint16_t* base, std::uint16_t* y, int rows,
                               int conv_stream, bool out_bf16 = false) {
        if (out_bf16) {
            dflash_dyn_conv_kernel<DFlashFmt::BF16><<<dim3(rows, kHidden / 256), 256, 0, stream>>>(x, dyn, base, y, rows, conv_stream);
        } else {
            dflash_dyn_conv_kernel<DFlashFmt::F16><<<dim3(rows, kHidden / 256), 256, 0, stream>>>(x, dyn, base, y, rows, conv_stream);
        }
        cuda_check(cudaGetLastError(), "E5A2 launch draft dynamic conv");
    };
    auto launch_residual = [&](const std::uint16_t* a, const std::uint16_t* b,
                               std::uint16_t* out, int count) {
        dflash_residual_kernel<<<(count + 255) / 256, 256, 0, stream>>>(a, b, out, count);
        cuda_check(cudaGetLastError(), "E5A2 launch draft residual");
    };
    auto launch_rope = [&](const std::uint16_t* in, std::uint16_t* out,
                           const std::int32_t* positions, int rows, int heads) {
        if(m.gaming[Gopt::DraftRopePair]) {
            static_assert(kRopeDim==kHeadDim);
            gopt_draft_rope_pair_kernel<<<(rows*heads*(kHeadDim/2)+255)/256,256,0,stream>>>(
                in,out,positions,rows,heads);
            cuda_check(cudaGetLastError(),"launch GOPT draft paired RoPE");
            gopt_record(m.gaming_submissions,Gopt::DraftRopePair);
        } else dflash_rope_kernel<<<(rows * heads * kHeadDim + 255) / 256, 256, 0, stream>>>(
            in, out, positions, rows, heads, kHeadDim);
        cuda_check(cudaGetLastError(), "E5A2 launch draft RoPE");
    };

    // E5A2 diagnostic stage tracing (env-gated, no effect on results).
    std::unique_ptr<DeviceAllocation> nan_alloc;
    unsigned long long* nan_count = nullptr;
    static bool g_traced_once = false;
    const bool stage_trace = (std::getenv("NINFER_DFLASH2_DEBUG_LOGITS") != nullptr) &&
                             !g_traced_once;
    if (stage_trace) {
        g_traced_once = true;
        nan_alloc = std::make_unique<DeviceAllocation>(sizeof(unsigned long long),
                                                       "E5A2 debug nan counter");
        nan_count = static_cast<unsigned long long*>(nan_alloc->ptr);
    }
    bool trace_hit = false;
    auto stage_maxabs = [&](const std::string& label, const std::uint16_t* buf, int count, bool is_bf16 = false) {
        if (!stage_trace) return;
        std::vector<std::uint16_t> host(static_cast<std::size_t>(count));
        cuda_check(cudaMemcpyAsync(host.data(), buf, host.size() * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToHost, stream), "E5A2 maxabs download");
        cuda_check(cudaStreamSynchronize(stream), "E5A2 maxabs sync");
        float maxabs = 0.0f;
        std::size_t over30k = 0, over60k = 0;
        for (const auto bits : host) {
            const float value = std::fabs(dflash_load(bits, is_bf16 ? DFlashFmt::BF16 : DFlashFmt::F16));
            if (value != value) maxabs = std::numeric_limits<float>::quiet_NaN();
            maxabs = std::max(maxabs, value);
            if (value > 30000.0f) ++over30k;
            if (value > 60000.0f) ++over60k;
        }
        std::cout << "E5A2 maxabs " << label << " max=" << maxabs
                  << " over30k=" << over30k << " over60k=" << over60k << '\n';
    };
    auto stage_nan = [&](const std::string& label, const std::uint16_t* buf, int count, bool is_bf16 = false) {
        if (!stage_trace || trace_hit) return;
        cuda_check(cudaMemsetAsync(nan_count, 0, sizeof(unsigned long long), stream),
                   "E5A2 nan zero");
        dflash_count_nonfinite_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
            buf, count, nan_count, is_bf16 ? 1 : 0);
        cuda_check(cudaGetLastError(), "E5A2 nan count launch");
        unsigned long long value = 0;
        cuda_check(cudaMemcpyAsync(&value, nan_count, sizeof(value),
                                   cudaMemcpyDeviceToHost, stream),
                   "E5A2 nan read");
        cuda_check(cudaStreamSynchronize(stream), "E5A2 nan sync");
        if (value > 0) {
            std::cout << "E5A2 stage_nan " << label << " nonfinite=" << value << "/" << count << '\n';
            trace_hit = true;
        }
    };

    // Block embedding (target embed_tokens, BF16 -> BF16 residual storage) into layer state A.
    dflash_embed_kernel<<<(block_len * kHidden + 255) / 256, 256, 0, stream>>>(
        m.s.ids, target_embedding_bf16, m.s.x_a, block_len);
    cuda_check(cudaGetLastError(), "E5A2 launch draft embedding");
    stage_nan("embed_xa", m.s.x_a, block_len * kHidden, true);
    stage_maxabs("embed_xa", m.s.x_a, block_len * kHidden, true);
    ph_end("embed");
    ph_begin();

    // Context hidden states from the target taps: concat -> fc -> hidden_norm.
    // E5A3: skipped on the ring path (committed K/V are already in the ring).
    if (!use_ring) {
    const int concat_total = window_rows * kLayers * kHidden;
    dflash_tap_concat_kernel<<<(concat_total + 255) / 256, 256, 0, stream>>>(
        m.s.tap_ptr, m.s.tcat, window_rows);
    cuda_check(cudaGetLastError(), "E5A2 launch draft tap concat");
    timing_close(Impl::TimingCategory::DenseMisc, "setup_and_tap_concat");
    timed_forward(*m.ws_fc, m.fc, m.fc_meta, m.s.tcat, m.s.fc_raw,
                  window_rows, -1, "fc", Impl::TimingCategory::OtherProjection);
    stage_nan("fc_raw", m.s.fc_raw, window_rows * kHidden);
    launch_rms(m.s.fc_raw, m.hidden_norm, m.s.hctx, window_rows, kHidden);
    stage_nan("hctx", m.s.hctx, window_rows * kHidden);
    stage_maxabs("hctx", m.s.hctx, window_rows * kHidden);
    }  // !use_ring: ring path reads committed K/V, no tap recompute
    ph_end("tap_fc");
    ph_begin();  // pre-forward (gated ref32 + setup, host-side)
    // E5A2 FP32 independent reference (env-gated, one-shot, read-only).
    // Downloads the real block embeddings + tap concat and replays the draft in
    // FP32 storage with the same EXL3 oracle. Decides FP16-range vs decode bug.
    // No effect on the device path.
    static bool g_ref32_done = false;
    if (!use_ring && !g_ref32_done && std::getenv("NINFER_DFLASH2_REF_FP32") != nullptr) {
        g_ref32_done = true;
        auto ref32_grab = [&](const std::uint16_t* d, std::size_t n) {
            std::vector<std::uint16_t> h(n);
            cuda_check(cudaMemcpyAsync(h.data(), d, n * sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToHost, stream),
                       "E5A2 ref32 download");
            cuda_check(cudaStreamSynchronize(stream), "E5A2 ref32 sync");
            return h;
        };
        const auto xa_h = ref32_grab(m.s.x_a, static_cast<std::size_t>(block_len) * kHidden);
        const auto tcat_h = ref32_grab(m.s.tcat, static_cast<std::size_t>(window_rows) * kLayers * kHidden);
        ref32_forward(m.directory, m.header, xa_h, tcat_h, host_pos_ctx, host_pos_blk,
                      block_len, window_rows);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    uint16_t* current = m.s.x_a;
    uint16_t* other = m.s.x_b;

    // E5A2 first-divergence differential (env-gated, one-shot). Recomputes the
    // layer-0 o_proj module (4096->5120, K5, mul1) on the host with the qualified
    // EXL3 decode convention on the IDENTICAL real attention output, then compares
    // the CUDA workspace output at the failing M and at M=1/2. The workspace output
    // is restored to the full row count afterwards so the live forward is untouched.
    static bool g_l0_diff_done = false;
    auto diff_l0_oproj = [&](int m_rows) {
        if (g_l0_diff_done) return;
        if (std::getenv("NINFER_DFLASH2_DIFF_L0_OPROJ") == nullptr) return;
        g_l0_diff_done = true;
        const int in = kQHeads * kHeadDim;   // 4096
        const int out = kHidden;             // 5120
        const auto& w = m.layers[0].o;
        const auto& md = m.layers[0].om;
        std::vector<std::uint16_t> attn_h((size_t)m_rows * in);
        std::vector<std::uint16_t> oproj_dev((size_t)m_rows * out);
        cuda_check(cudaMemcpyAsync(attn_h.data(), m.s.attn,
                                   attn_h.size() * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToHost, stream),
                   "E5A2 diff attn download");
        cuda_check(cudaMemcpyAsync(oproj_dev.data(), m.s.oproj,
                                   oproj_dev.size() * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToHost, stream),
                   "E5A2 diff oproj download");
        cuda_check(cudaStreamSynchronize(stream), "E5A2 diff sync");
        const auto host_mod = read_host_linear(m.directory, m.header,
                                               "layers.0.self_attn.o_proj", in, out);
        const auto host_out = host_linear_forward(host_mod, attn_h.data(), m_rows);
        auto report = [&](const char* tag, int mm,
                          const std::vector<std::uint16_t>& dev) {
            double sq_err = 0.0, sq_host = 0.0;
            double max_abs_diff = 0.0, dev_max = 0.0, host_max = 0.0;
            std::size_t dev_nf = 0, host_nf = 0;
            int first_bad_row = -1, first_bad_col = -1;
            double first_bad_mag = 0.0;
            std::vector<double> row_max(mm, 0.0);
            for (int row = 0; row < mm; ++row) {
                for (int c = 0; c < out; ++c) {
                    const std::size_t idx = (size_t)row * out + c;
                    const float dv = half_to_float(dev[idx]);
                    const float hv = half_to_float(host_out[idx]);
                    if (!std::isfinite(dv)) ++dev_nf;
                    if (!std::isfinite(hv)) ++host_nf;
                    const double de = (double)dv - hv;
                    dev_max = std::max(dev_max, (double)std::fabs(dv));
                    host_max = std::max(host_max, (double)std::fabs(hv));
                    sq_err += de * de;
                    sq_host += (double)hv * hv;
                    const double ad = std::fabs(de);
                    max_abs_diff = std::max(max_abs_diff, ad);
                    row_max[row] = std::max(row_max[row], ad);
                    if (first_bad_row < 0 && ad > 1.0e-2 * (1.0 + std::fabs(hv))) {
                        first_bad_row = row; first_bad_col = c; first_bad_mag = ad;
                    }
                }
            }
            const double rel = sq_host > 0.0 ? std::sqrt(sq_err / sq_host) : std::sqrt(sq_err);
            double host_rms = std::sqrt(sq_host / (mm * out));
            std::vector<double> in_rms(mm, 0.0), in_mean(mm, 0.0);
            for (int row = 0; row < mm; ++row) {
                double s2 = 0.0, s1 = 0.0;
                for (int c = 0; c < in; ++c) {
                    const float v = half_to_float(attn_h[(size_t)row * in + c]);
                    s2 += (double)v * v; s1 += v;
                }
                in_rms[row] = std::sqrt(s2 / in);
                in_mean[row] = s1 / in;
            }
            std::cout << "E5A2 diff_l0_oproj tag=" << tag << " M=" << mm
                      << " dev_max=" << dev_max << " host_max=" << host_max
                      << " host_rms=" << host_rms
                      << " max_abs_diff=" << max_abs_diff
                      << " rel_l2=" << rel << " rms_err=" << std::sqrt(sq_err / (mm * out))
                      << " dev_nonfinite=" << dev_nf << " host_nonfinite=" << host_nf
                      << " first_div_row=" << first_bad_row
                      << " first_div_col=" << first_bad_col
                      << " first_div_mag=" << first_bad_mag << '\n';
            std::cout << "E5A2 diff_l0_oproj_io M=" << mm
                      << " in_row0_rms=" << in_rms[0] << " in_row0_mean=" << in_mean[0];
            for (int row = 1; row < mm; ++row)
                std::cout << " in_r" << row << "=" << in_rms[row];
            std::cout << '\n';
        };
        report("device_M_full", m_rows, oproj_dev);
        // M=1 and M=2 use the same real input/weights to separate geometry from M>1.
        for (int mm : {1, 2}) {
            if (mm >= m_rows) continue;
            m.ws_o->forward(w, md, m.s.attn, m.s.oproj, mm, stream);
            std::vector<std::uint16_t> dev_mm((size_t)mm * out);
            cuda_check(cudaMemcpyAsync(dev_mm.data(), m.s.oproj,
                                       dev_mm.size() * sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToHost, stream),
                       "E5A2 diff M download");
            cuda_check(cudaStreamSynchronize(stream), "E5A2 diff M sync");
            report(("device_M" + std::to_string(mm)).c_str(), mm, dev_mm);
        }
        // Restore the live full-row o_proj output for the continuing forward.
        m.ws_o->forward(w, md, m.s.attn, m.s.oproj, m_rows, stream);
    };

    // E5A2 attention-stage differential (env-gated, one-shot): host fp32 softmax
    // attention over the SAME post-RoPE q/k and V rows the device kernel consumed,
    // compared element-wise against the device attention output.
    static bool g_l0_attn_done = false;
    auto diff_l0_attention = [&](int q_rows, int ctx_rows) {
        if (g_l0_attn_done) return;
        if (std::getenv("NINFER_DFLASH2_DIFF_L0_ATTN") == nullptr) return;
        g_l0_attn_done = true;
        auto grab = [&](const std::uint16_t* d, std::size_t n) {
            std::vector<std::uint16_t> h(n);
            cuda_check(cudaMemcpyAsync(h.data(), d, n * sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToHost, stream),
                       "E5A2 attn diff download");
            cuda_check(cudaStreamSynchronize(stream), "E5A2 attn diff sync");
            return h;
        };
        const std::size_t qn = (size_t)q_rows * kQHeads * kHeadDim;       // 4096
        const std::size_t kvn = (size_t)q_rows * kKVHeads * kHeadDim;     // 1024 (block)
        const std::size_t kvcn = (size_t)ctx_rows * kKVHeads * kHeadDim;  // 1024 (ctx)
        const auto qr = grab(m.s.qr, qn);
        const auto kctx = grab(m.s.kctx_r, kvcn);
        const auto vctx = grab(m.s.vctx, kvcn);
        const auto kblk = grab(m.s.kblk_r, kvn);
        const auto vblk = grab(m.s.vblk, kvn);
        const auto attn_dev = grab(m.s.attn, qn);
        const int keys = ctx_rows + q_rows;
        std::vector<std::uint16_t> attn_host(qn, 0);
        double sq_err = 0.0, sq_dev = 0.0, max_diff = 0.0, dev_max = 0.0, host_max = 0.0;
        std::size_t nf = 0;
        std::array<double, 64> scores{};
        const float fscale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
        for (int query = 0; query < q_rows; ++query) {
            for (int head = 0; head < kQHeads; ++head) {
                const int kv_head = head / (kQHeads / kKVHeads);
                const std::uint16_t* q_row = qr.data() + (size_t)(query * kQHeads + head) * kHeadDim;
                double maximum = -1.0e30;
                for (int key = 0; key < keys; ++key) {
                    const std::uint16_t* k_row = key < ctx_rows
                        ? kctx.data() + (size_t)(key * kKVHeads + kv_head) * kHeadDim
                        : kblk.data() + (size_t)((key - ctx_rows) * kKVHeads + kv_head) * kHeadDim;
                    double dot = 0.0;
                    for (int d = 0; d < kHeadDim; ++d)
                        dot += (double)half_to_float(q_row[d]) * half_to_float(k_row[d]);
                    scores[key] = dot * fscale;
                    maximum = std::max(maximum, scores[key]);
                }
                double denominator = 0.0;
                for (int key = 0; key < keys; ++key) {
                    scores[key] = std::exp(scores[key] - maximum);
                    denominator += scores[key];
                }
                for (int d = 0; d < kHeadDim; ++d) {
                    double value = 0.0;
                    for (int key = 0; key < keys; ++key) {
                        const std::uint16_t* v_row = key < ctx_rows
                            ? vctx.data() + (size_t)(key * kKVHeads + kv_head) * kHeadDim
                            : vblk.data() + (size_t)((key - ctx_rows) * kKVHeads + kv_head) * kHeadDim;
                        value += scores[key] / denominator * half_to_float(v_row[d]);
                    }
                    const auto hv = float_to_half((float)value);
                    const std::size_t idx = (size_t)(query * kQHeads + head) * kHeadDim + d;
                    attn_host[idx] = hv;
                    const double dv = half_to_float(attn_dev[idx]);
                    const double hv2 = half_to_float(hv);
                    const double diff = dv - hv2;
                    if (!std::isfinite(dv) || !std::isfinite(hv2)) ++nf;
                    sq_err += diff * diff;
                    sq_dev += dv * dv;
                    max_diff = std::max(max_diff, std::fabs(diff));
                    dev_max = std::max(dev_max, std::fabs(dv));
                    host_max = std::max(host_max, std::fabs(hv2));
                }
            }
        }
        const double rel = sq_dev > 0.0 ? std::sqrt(sq_err / sq_dev) : std::sqrt(sq_err);
        std::cout << "E5A2 diff_l0_attn M=" << q_rows << " keys=" << keys
                  << " dev_max=" << dev_max << " host_max=" << host_max
                  << " max_abs_diff=" << max_diff << " rel_l2=" << rel
                  << " nonfinite=" << nf << '\n';
    };

    // E5A2 conv-finish differential (env-gated, one-shot). Host fp32 recompute of the
    // grouped dynamic conv finish over the SAME real input/dyn/base rows the device
    // consumed, compared to the device convf output. stream=1 is the finish stream.
    static int g_l0_conv_count = 0;
    auto diff_l0_conv_finish = [&](int rows, const std::uint16_t* input_dev,
                                   const std::uint16_t* dyn_dev,
                                   const std::uint16_t* base_dev,
                                   const std::uint16_t* out_dev,
                                   const std::string& tag) {
        if (g_l0_conv_count >= 2) return;
        if (std::getenv("NINFER_DFLASH2_DIFF_L0_CONV") == nullptr) return;
        ++g_l0_conv_count;
        auto grab = [&](const std::uint16_t* d, std::size_t n) {
            std::vector<std::uint16_t> h(n);
            cuda_check(cudaMemcpyAsync(h.data(), d, n * sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToHost, stream),
                       "E5A2 conv diff download");
            cuda_check(cudaStreamSynchronize(stream), "E5A2 conv diff sync");
            return h;
        };
        const std::size_t xn = (size_t)rows * kHidden;
        const std::size_t dynn = (size_t)rows * kConvDynamic;
        const std::size_t basen = 2u * kConvKernel * kHidden;
        const auto x = grab(input_dev, xn);
        const auto dyn = grab(dyn_dev, dynn);
        const auto base = grab(base_dev, basen);
        const auto out_dev_h = grab(out_dev, xn);
        std::vector<std::uint16_t> out_host(xn, 0);
        double sq_err = 0.0, sq_dev = 0.0, max_diff = 0.0, dev_max = 0.0, host_max = 0.0;
        std::size_t nf = 0;
        // stream s = 1: base[s][off][i], dyn[row][s*640 + off*320 + g]
        const int s = 1;
        for (int row = 0; row < rows; ++row) {
            for (int i = 0; i < kHidden; ++i) {
                const int g = i / kConvGroup;
                const float x0 = half_to_float(x[(size_t)row * kHidden + i]);
                const float x1 = row > 0 ? half_to_float(x[(size_t)(row - 1) * kHidden + i]) : 0.0f;
                const float d0 = half_to_float(dyn[(size_t)row * kConvDynamic + s * (kConvKernel * kConvGroups) + g]);
                const float d1 = half_to_float(dyn[(size_t)row * kConvDynamic + s * (kConvKernel * kConvGroups) + kConvGroups + g]);
                const std::uint16_t* brow = base.data() + s * (kConvKernel * kHidden);
                const float b0 = half_to_float(brow[i]);
                const float b1 = half_to_float(brow[kHidden + i]);
                const auto hv = float_to_bf16((b0 + d0) * x0 + (b1 + d1) * x1);  // device stores BF16
                const std::size_t idx = (size_t)row * kHidden + i;
                out_host[idx] = hv;
                const double dv = bf16_to_float(out_dev_h[idx]);
                const double hv2 = bf16_to_float(hv);
                const double diff = dv - hv2;
                if (!std::isfinite(dv) || !std::isfinite(hv2)) ++nf;
                sq_err += diff * diff;
                sq_dev += dv * dv;
                max_diff = std::max(max_diff, std::fabs(diff));
                dev_max = std::max(dev_max, std::fabs(dv));
                host_max = std::max(host_max, std::fabs(hv2));
            }
        }
        const double rel = sq_dev > 0.0 ? std::sqrt(sq_err / sq_dev) : std::sqrt(sq_err);
        std::cout << "E5A2 diff_l0_conv tag=" << tag << " M=" << rows
                  << " dev_max=" << dev_max << " host_max=" << host_max
                  << " max_abs_diff=" << max_diff << " rel_l2=" << rel
                  << " nonfinite=" << nf << '\n';
    };
    ph_end("prefwd");
    ph_begin();  // layer0 ctxkv
for (int layer_index = 0; layer_index < kLayers; ++layer_index) {
        const auto& lay = m.layers[layer_index];

        // Context K/V (same normalized hctx for every draft layer).
        // E5A3: skipped on the ring path (committed K/V already in the ring).
        if (!use_ring) {
        timing_close(Impl::TimingCategory::DenseMisc, "context_setup");
        timed_forward(*m.ws_kv, lay.k, lay.km, m.s.hctx, m.s.kctx,
                      window_rows, layer_index, "context_k",
                      Impl::TimingCategory::OtherProjection);
        timed_forward(*m.ws_kv, lay.v, lay.vm, m.s.hctx, m.s.vctx,
                      window_rows, layer_index, "context_v",
                      Impl::TimingCategory::OtherProjection);
        launch_head_norm(m.s.kctx, lay.k_norm, m.s.kctx_n,
                         window_rows * kKVHeads, kHeadDim);
        launch_rope(m.s.kctx_n, m.s.kctx_r, m.s.pos_ctx, window_rows, kKVHeads);
        }  // !use_ring
        ph_end("layer" + std::to_string(layer_index) + "_ctxkv");
        ph_begin();  // block-qkv + attention phase
        stage_maxabs("vctx_" + std::to_string(layer_index), m.s.vctx, window_rows * kKVHeads * kHeadDim);

        // Block normalized input, kernel projection, conv stream 0.
        launch_rms(current, lay.input_norm, m.s.ln, block_len, kHidden, true);
        stage_nan("ln_" + std::to_string(layer_index), m.s.ln, block_len * kHidden);
        if (use_ring) timing_close(Impl::TimingCategory::DenseMisc, "attn_norm");
        launch_dense(m.s.ln, lay.attn_kernel_proj, m.s.dyn, block_len, kHidden,
                     kConvDynamic);
        if (use_ring) timing_close(Impl::TimingCategory::DenseMisc, "attn_dense_projection");
        launch_dyn_conv(m.s.ln, m.s.dyn, lay.attn_base, m.s.conv, block_len, 0);
        stage_nan("conv_" + std::to_string(layer_index), m.s.conv, block_len * kHidden);
        timing_close(Impl::TimingCategory::DenseMisc,
                     use_ring ? "attn_dynconv" : "attn_norm_dense_dynconv");

        // Block Q/K/V projections from the convolved input.
        timed_forward(*m.ws_q, lay.q, lay.qm, m.s.conv, m.s.qproj,
                      block_len, layer_index, "q", Impl::TimingCategory::Projection);
        timed_forward(*m.ws_kv, lay.k, lay.km, m.s.conv, m.s.kblk,
                      block_len, layer_index, "k", Impl::TimingCategory::Projection);
        timed_forward(*m.ws_kv, lay.v, lay.vm, m.s.conv, m.s.vblk,
                      block_len, layer_index, "v", Impl::TimingCategory::Projection);
        stage_maxabs("vblk_" + std::to_string(layer_index), m.s.vblk, block_len * kKVHeads * kHeadDim);
        if(m.gaming[Gopt::DraftHeadNormRope]) {
            static_assert(kHeadDim==128 && kRopeDim==128);
            const bool paired=m.gaming[Gopt::DraftRopePair];
            dflash_head_norm_kernel<true><<<block_len*kQHeads,128,128*sizeof(float),stream>>>(
                m.s.qproj,lay.q_norm,m.s.qn,block_len*kQHeads,kHeadDim,m.s.pos_blk,m.s.qr,kQHeads,paired);
            cuda_check(cudaGetLastError(),"launch GOPT draft Q norm/RoPE");
            dflash_head_norm_kernel<true><<<block_len*kKVHeads,128,128*sizeof(float),stream>>>(
                m.s.kblk,lay.k_norm,m.s.kblk_n,block_len*kKVHeads,kHeadDim,m.s.pos_blk,m.s.kblk_r,kKVHeads,paired);
            cuda_check(cudaGetLastError(),"launch GOPT draft K norm/RoPE");
            gopt_record(m.gaming_submissions,Gopt::DraftHeadNormRope);
            if(paired)gopt_record(m.gaming_submissions,Gopt::DraftRopePair);
        } else {
            launch_head_norm(m.s.qproj, lay.q_norm, m.s.qn, block_len * kQHeads, kHeadDim);
            launch_head_norm(m.s.kblk, lay.k_norm, m.s.kblk_n, block_len * kKVHeads, kHeadDim);
            launch_rope(m.s.qn, m.s.qr, m.s.pos_blk, block_len, kQHeads);
            launch_rope(m.s.kblk_n, m.s.kblk_r, m.s.pos_blk, block_len, kKVHeads);
        }

        // Attention over the S context keys + L block keys (non-causal block).
        if (use_ring) {
            // E5A3: identical math over ring-committed ctx keys (online softmax,
            // modulo indexing; all committed keys within the 2048 window).
            // A shared projection never combines attention histories. When the
            // B8 sharing scope exists, consume its lane-private mask/ring view;
            // the peer's segment is deliberately unavailable in this caller.
            if(shared_segment_enabled)
                require(shared_segment.attention_view_matches(m.ring_base_abs,m.ring_count,
                    block_pos0,block_len),"shared draft private attention view changed");
            const int ring_start = shared_segment_enabled?shared_segment.ring_start_slot():
                static_cast<int>(m.ring_base_abs & kRingMask);
            const int private_ring_count=shared_segment_enabled?shared_segment.ring_count:m.ring_count;
            if (m.parallel_ring_attention && dflash_ring_split_selected(private_ring_count))
                launch_ring_attention_split(m.s.qr, m.ring_k[layer_index], m.ring_v[layer_index],
                    ring_start, private_ring_count, m.s.kblk_r, m.s.vblk, m.s.attn, block_len,
                    block_len, scale, m.s.ring_partial_o, m.s.ring_partial_ml, stream);
            else if (m.parallel_ring_attention && dflash_ring_staged_enabled())
                dflash_attention_ring_staged_kernel<<<block_len * kQHeads, kHeadDim, 0, stream>>>(
                    m.s.qr, m.ring_k[layer_index], m.ring_v[layer_index], ring_start,
                    private_ring_count, m.s.kblk_r, m.s.vblk, m.s.attn, block_len, block_len, scale);
            else if (m.parallel_ring_attention)
                dflash_attention_ring_parallel_kernel<<<block_len * kQHeads, kHeadDim, 0, stream>>>(
                    m.s.qr, m.ring_k[layer_index], m.ring_v[layer_index], ring_start,
                    private_ring_count, m.s.kblk_r, m.s.vblk, m.s.attn, block_len, block_len, scale);
            else
                dflash_attention_ring_kernel<<<block_len * kQHeads, 1, 0, stream>>>(
                    m.s.qr, m.ring_k[layer_index], m.ring_v[layer_index], ring_start,
                    private_ring_count, m.s.kblk_r, m.s.vblk, m.s.attn, block_len, block_len, scale);
            cuda_check(cudaGetLastError(), "launch ring attention");
            if (m.ring_attention_observer) {
                RingAttentionObservation observation{m.s.qr, m.ring_k[layer_index],
                    m.ring_v[layer_index], m.s.kblk_r, m.s.vblk, m.s.attn,
                    ring_start, private_ring_count, block_len, block_len, layer_index, scale, stream};
                m.ring_attention_observer(observation, m.ring_attention_observer_user);
            }
        } else {
        dflash_attention_kernel<<<block_len * kQHeads, 1, 0, stream>>>(
            m.s.qr, m.s.kctx_r, m.s.vctx, m.s.kblk_r, m.s.vblk, m.s.attn,
            block_len, window_rows, block_len, scale);
        }
        cuda_check(cudaGetLastError(), "E5A2 launch draft attention");
        stage_maxabs("attnout_" + std::to_string(layer_index), m.s.attn, block_len * kQHeads * kHeadDim);
        if (!use_ring && layer_index == 0) diff_l0_attention(block_len, window_rows);
        timing_close(Impl::TimingCategory::Attention,
                     use_ring ? "ring_attention" : "window_attention");
        ph_end("layer" + std::to_string(layer_index) + "_attn");
        ph_begin();  // o_proj + conv + mlp phase

        timed_forward(*m.ws_o, lay.o, lay.om, m.s.attn, m.s.oproj,
                      block_len, layer_index, "o", Impl::TimingCategory::Projection);
        stage_nan("attno_" + std::to_string(layer_index), m.s.oproj, block_len * kHidden);
        stage_maxabs("oproj_" + std::to_string(layer_index), m.s.oproj, block_len * kHidden);
        if (!use_ring && layer_index == 0) diff_l0_oproj(block_len);
        if(m.gaming[Gopt::DraftConvResidual]) {
            dflash_dyn_conv_kernel<DFlashFmt::BF16,true><<<dim3(block_len, kHidden / 256), 256,0,stream>>>(
                m.s.oproj,m.s.dyn,lay.attn_base,m.s.convf,block_len,1,current,other);
            cuda_check(cudaGetLastError(),"launch GOPT draft conv/residual");
            gopt_record(m.gaming_submissions,Gopt::DraftConvResidual);
        } else launch_dyn_conv(m.s.oproj, m.s.dyn, lay.attn_base, m.s.convf, block_len, 1, true);
        if (!use_ring && layer_index == 0)
            diff_l0_conv_finish(block_len, m.s.oproj, m.s.dyn, lay.attn_base,
                                m.s.convf, "attn_convf");
        stage_maxabs("attn_convf_" + std::to_string(layer_index), m.s.convf, block_len * kHidden, true);
        if(!m.gaming[Gopt::DraftConvResidual] && !m.gaming[Gopt::DraftResidualNorm])
            launch_residual(current, m.s.convf, other, block_len * kHidden);
        if(m.gaming[Gopt::DraftResidualNorm]) {
            dflash_rms_norm_kernel<DFlashFmt::BF16,true><<<block_len,256,256*sizeof(float),stream>>>(
                current,lay.post_norm,m.s.ln2,block_len,kHidden,m.s.convf,other);
            cuda_check(cudaGetLastError(),"launch GOPT draft residual/norm");
            gopt_record(m.gaming_submissions,Gopt::DraftResidualNorm);
        }
        stage_nan("residA_" + std::to_string(layer_index), other, block_len * kHidden, true);
        stage_maxabs("residA_" + std::to_string(layer_index), other, block_len * kHidden, true);

        // MLP with its own dynamic conv.
        if(!m.gaming[Gopt::DraftResidualNorm])
            launch_rms(other, lay.post_norm, m.s.ln2, block_len, kHidden, true);
        stage_nan("mlp_ln2_" + std::to_string(layer_index), m.s.ln2, block_len * kHidden);
        if (use_ring) timing_close(Impl::TimingCategory::DenseMisc,
                                   "attn_finish_residual_mlp_norm");
        launch_dense(m.s.ln2, lay.mlp_kernel_proj, m.s.dyn2, block_len, kHidden,
                     kConvDynamic);
        if (use_ring) timing_close(Impl::TimingCategory::DenseMisc, "mlp_dense_projection");
        launch_dyn_conv(m.s.ln2, m.s.dyn2, lay.mlp_base, m.s.conv2, block_len, 0);
        stage_nan("mlp_conv2_" + std::to_string(layer_index), m.s.conv2, block_len * kHidden);
        timing_close(Impl::TimingCategory::DenseMisc,
                     use_ring ? "mlp_dynconv" :
                         "attn_finish_residual_mlp_norm_dense_dynconv");
        {
        // conv2 is immutable through these two synchronous projection calls.
        // No lifetime atomics are touched on the default independent route.
        std::optional<Exl3ActivationLifetime::Scope> gateup_lifetime;
        if(shared_segment_enabled && m.shared_block_gateup) {
            gateup_lifetime.emplace(m.shared_mlp_activation_lifetime);
            shared_mlp_activation=gateup_lifetime->witness();
        }
        timed_forward(*m.ws_mlp, lay.gate, lay.gm, m.s.conv2, m.s.gate,
                      block_len, layer_index, "gate", Impl::TimingCategory::Projection);
        timed_forward(*m.ws_mlp, lay.up, lay.um, m.s.conv2, m.s.up,
                      block_len, layer_index, "up", Impl::TimingCategory::Projection);
        shared_mlp_activation={};
        } // Expire the activation even when a projection callback throws.
        stage_nan("mlp_gate_" + std::to_string(layer_index), m.s.gate, block_len * kIntermediate);
        stage_nan("mlp_up_" + std::to_string(layer_index), m.s.up, block_len * kIntermediate);
        stage_maxabs("mlp_gate_" + std::to_string(layer_index), m.s.gate, block_len * kIntermediate);
        stage_maxabs("mlp_up_" + std::to_string(layer_index), m.s.up, block_len * kIntermediate);
        dflash_silu_mul_kernel<<<(block_len * kIntermediate + 255) / 256, 256, 0, stream>>>(
            m.s.gate, m.s.up, m.s.act, block_len * kIntermediate);
        cuda_check(cudaGetLastError(), "E5A2 launch draft silu-mul");
        stage_nan("mlp_act_" + std::to_string(layer_index), m.s.act, block_len * kIntermediate);
        stage_maxabs("mlp_act_" + std::to_string(layer_index), m.s.act, block_len * kIntermediate);
        timing_close(Impl::TimingCategory::DenseMisc, "mlp_activation");
        timed_forward(*m.ws_down, lay.down, lay.dm, m.s.act, m.s.down,
                      block_len, layer_index, "down", Impl::TimingCategory::Projection);
        stage_nan("mlp_down_" + std::to_string(layer_index), m.s.down, block_len * kHidden);
        stage_maxabs("mlp_down_" + std::to_string(layer_index), m.s.down, block_len * kHidden);
        if(m.gaming[Gopt::DraftConvResidual]) {
            dflash_dyn_conv_kernel<DFlashFmt::BF16,true><<<dim3(block_len, kHidden / 256), 256,0,stream>>>(
                m.s.down,m.s.dyn2,lay.mlp_base,m.s.convf2,block_len,1,other,other);
            cuda_check(cudaGetLastError(),"launch GOPT draft conv/residual");
            gopt_record(m.gaming_submissions,Gopt::DraftConvResidual);
        } else launch_dyn_conv(m.s.down, m.s.dyn2, lay.mlp_base, m.s.convf2, block_len, 1, true);
        if (!use_ring && layer_index == 0)
            diff_l0_conv_finish(block_len, m.s.down, m.s.dyn2, lay.mlp_base,
                                m.s.convf2, "mlp_convf2");
        stage_nan("mlp_convf2_" + std::to_string(layer_index), m.s.convf2, block_len * kHidden, true);
        if(!m.gaming[Gopt::DraftConvResidual])
            launch_residual(other, m.s.convf2, other, block_len * kHidden);
        stage_nan("residM_" + std::to_string(layer_index), other, block_len * kHidden, true);
        stage_maxabs("residM_" + std::to_string(layer_index), other, block_len * kHidden, true);
        timing_close(Impl::TimingCategory::DenseMisc, "mlp_finish_residual");

        std::swap(current, other);
        ph_end("layer" + std::to_string(layer_index) + "_rest");
        ph_begin();  // next-layer ctxkv (or final head after last layer)
    }

    // Final draft RMSNorm over the speculative rows (logits_start == 1 drops the
    // block anchor row), then the TARGET H6 head computes proposal logits.
    stage_nan("pre_final", current + kHidden, proposal_rows * kHidden, true);
    launch_rms(current + kHidden, m.norm, m.s.final_norm, proposal_rows, kHidden, true);
    stage_nan("final_norm", m.s.final_norm, proposal_rows * kHidden);
    timing_close(Impl::TimingCategory::DenseMisc, "final_norm");
    timed_forward(*m.ws_head, target_head, target_head_metadata, m.s.final_norm,
                  m.s.head_out, proposal_rows, -1, "target_h6",
                  Impl::TimingCategory::H6);
    launch_dense(m.s.final_norm, m.hidden_proj, m.s.hidden_proj_out, proposal_rows,
                 kHidden, kRank);

    // Candidate selector: top-16 by unary, then predecessor codebook edge scores.
    const bool fused_selector = m.fused_selector && !m.position_confidence &&
        !projection_timing.enabled && (!device_seed || proposal_rows <= 7);
    const bool fused_liveness = m.fused_topk_liveness && fused_selector;
    if (fused_liveness) {
        require(m.s.liveness_flag != nullptr,
                "fused top-K liveness flag allocation missing");
        cuda_check(cudaMemsetAsync(m.s.liveness_flag, 0,
                                   sizeof(unsigned int), stream),
                   "zero fused top-K liveness flag");
    }
    if (m.local_merge_topk) {
        if (fused_liveness) {
            dflash_topk16_local_merge_kernel<true><<<proposal_rows, 256, 0, stream>>>(
                m.s.head_out, proposal_rows, kVocab, m.s.cand_ids,
                m.s.cand_unary, m.s.liveness_flag);
            ++m.fused_topk_liveness_calls;
        } else if (dflash_topk_segmented_enabled()) {
            launch_dflash_topk16_segmented(m.s.head_out, proposal_rows, kVocab,
                m.s.cand_ids, m.s.cand_unary, m.s.topk_segments, stream);
        } else {
            dflash_topk16_local_merge_kernel<false><<<proposal_rows, 256, 0, stream>>>(
                m.s.head_out, proposal_rows, kVocab, m.s.cand_ids,
                m.s.cand_unary, nullptr);
        }
        ++m.local_merge_topk_calls;
    } else if (m.parallel_topk)
        dflash_topk16_parallel_kernel<<<proposal_rows, 256, 0, stream>>>(
            m.s.head_out, proposal_rows, kVocab, m.s.cand_ids, m.s.cand_unary);
    else
        dflash_topk16_kernel<<<proposal_rows, 1, 0, stream>>>(
            m.s.head_out, proposal_rows, kVocab, m.s.cand_ids, m.s.cand_unary);
    cuda_check(cudaGetLastError(), "E5A2 launch draft top-16");
    timing_close(Impl::TimingCategory::Selector, "selector_prepare");
    ph_end("final_head");
    ph_begin();  // liveness-guard phase

    // E5A2 diagnostic (env-gated, no effect on gates): report per-row head-logit
    // finiteness/spread so a dead/zero or NaN draft path is distinguishable from a
    // live-but-degenerate one. Not part of the forced-rejection contract.
    if (std::getenv("NINFER_DFLASH2_DEBUG_LOGITS") != nullptr) {
        std::vector<std::uint16_t> raw_logits(
            static_cast<std::size_t>(proposal_rows) * kVocab);
        cuda_check(cudaMemcpyAsync(raw_logits.data(), m.s.head_out,
                                   raw_logits.size() * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToHost, stream),
                   "E5A2 debug logits download");
        cuda_check(cudaStreamSynchronize(stream), "E5A2 debug logits sync");
        for (int r = 0; r < proposal_rows; ++r) {
            float minv = std::numeric_limits<float>::infinity();
            float maxv = -std::numeric_limits<float>::infinity();
            std::size_t finite = 0, nonzero = 0;
            for (int v = 0; v < kVocab; ++v) {
                const float val =
                    half_to_float(raw_logits[static_cast<std::size_t>(r) * kVocab + v]);
                if (!std::isfinite(val)) continue;
                ++finite;
                if (val != 0.0F) ++nonzero;
                minv = std::min(minv, val);
                maxv = std::max(maxv, val);
            }
            std::cout << "E5A2 dflash2_debug row=" << r
                      << " finite=" << finite << "/" << kVocab
                      << " nonzero=" << nonzero << " min=" << minv
                      << " max=" << maxv << '\n';
        }
    }

    // E5A2 fail-closed liveness guard: the FP16-overflow failure mode silently
    // produced token-0 proposals from non-finite logits. A non-finite head-logit
    // row must FAIL loudly instead of degenerating. FAST_DEVICE_LIVENESS keeps
    // the exact predicate and synchronization boundary but avoids materializing
    // the complete vocabulary rows on the host.
    unsigned int fused_bad = 0;
    if (m.fast_device_liveness) {
        require(m.s.liveness_flag != nullptr,
                "E5A2 fast liveness flag allocation missing");
        if (!fused_liveness) {
            const int count=proposal_rows*kVocab;
            cuda_check(cudaMemsetAsync(m.s.liveness_flag,0,sizeof(unsigned int),stream),
                       "E5A2 fast liveness zero");
            dflash_nonfinite_flag_kernel<<<(count+255)/256,256,0,stream>>>(
                m.s.head_out,count,m.s.liveness_flag);
            cuda_check(cudaGetLastError(),"E5A2 fast liveness launch");
        }
        cuda_check(cudaMemcpyAsync(&fused_bad,m.s.liveness_flag,sizeof(fused_bad),
                                   cudaMemcpyDeviceToHost,stream),
                   "E5A2 fast liveness read");
        if (!fused_liveness)
            cuda_check(cudaStreamSynchronize(stream),"E5A2 fast liveness sync");
        ++m.device_liveness_checks;
        if (!fused_liveness)
            require(fused_bad==0,
                "E5A2 draft head logits non-finite; proposals would be degenerate");
    } else {
        std::vector<std::uint16_t> logits_h(
            static_cast<std::size_t>(proposal_rows) * kVocab);
        cuda_check(cudaMemcpyAsync(logits_h.data(), m.s.head_out,
                                   logits_h.size() * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToHost, stream),
                   "E5A2 draft logits guard download");
        cuda_check(cudaStreamSynchronize(stream), "E5A2 draft logits guard sync");
        std::size_t bad = 0;
        for (const auto bits : logits_h) {
            if (!std::isfinite(half_to_float(bits))) ++bad;
        }
        require(bad == 0, "E5A2 draft head logits non-finite; proposals would be degenerate");
    }
    timing_close(Impl::TimingCategory::Selector, "liveness_guard_final_sync");

    ph_end("guard");
    ph_begin();  // selector phase
    std::vector<std::int64_t> result(static_cast<std::size_t>(proposal_rows));
    if (m.position_confidence)
        m.last_position_confidence.resize(static_cast<std::size_t>(proposal_rows));
    else
        m.last_position_confidence.clear();
    std::int64_t anchor = block_ids[0];
    require(anchor >= 0 && anchor < kVocab, "E5A2 draft anchor outside vocabulary");
    if (fused_selector) {
        dflash_selector_fused_chain_kernel<<<1, kSelectorFusedThreads, 0, stream>>>(
            proposal_rows, anchor, device_seed ? m.s.ids : nullptr,
            m.s.cand_ids, m.s.hidden_proj_out, m.pred_cb, m.succ_cb,
            m.s.cand_unary, m.s.proposal_out);
        cuda_check(cudaGetLastError(), "E5A2 launch fused draft selector");
        ++m.fused_selector_calls;
        cuda_check(cudaMemcpyAsync(result.data(), m.s.proposal_out,
                                   static_cast<std::size_t>(proposal_rows) *
                                       sizeof(std::int64_t),
                                   cudaMemcpyDeviceToHost, stream),
                   "E5A2 read fused draft proposals");
        timing_close(Impl::TimingCategory::Selector,
                     "selector_fused_final_sync");
        cuda_check(cudaStreamSynchronize(stream),
                   "E5A2 synchronize fused draft selector");
        if (fused_liveness)
            require(fused_bad==0,
                "E5A2 draft head logits non-finite; proposals would be degenerate");
        for (int p = 0; p < proposal_rows; ++p) {
            const auto selected = result[static_cast<std::size_t>(p)];
            require(selected >= 0 && selected < kVocab,
                    "E5A2 fused draft proposal outside vocabulary");
        }
    } else if (m.selector_batched_anchor_chain) {
        dflash_selector_batched_anchor_chain_kernel<<<
            1, kSelectorChainThreads, 0, stream>>>(
            proposal_rows, anchor, device_seed ? m.s.ids : nullptr,
            m.s.cand_ids, m.s.hidden_proj_out, m.pred_cb, m.succ_cb,
            m.s.cand_unary, m.s.edge_scores, m.s.proposal_out,
            m.position_confidence ? m.s.confidence_out : nullptr);
        cuda_check(cudaGetLastError(),
                   "E5A2 launch batched draft selector anchor chain");
        ++m.selector_batched_anchor_chain_calls;
        cuda_check(cudaMemcpyAsync(result.data(), m.s.proposal_out,
                                   static_cast<std::size_t>(proposal_rows) *
                                       sizeof(std::int64_t),
                                   cudaMemcpyDeviceToHost, stream),
                   "E5A2 read batched draft proposals");
        if (m.position_confidence)
            cuda_check(cudaMemcpyAsync(m.last_position_confidence.data(),
                                       m.s.confidence_out,
                                       static_cast<std::size_t>(proposal_rows) *
                                           sizeof(DraftPositionConfidence),
                                       cudaMemcpyDeviceToHost, stream),
                       "E5A2 read batched draft position confidence");
        timing_close(Impl::TimingCategory::Selector,
                     "selector_batched_final_sync");
        cuda_check(cudaStreamSynchronize(stream),
                   "E5A2 synchronize batched draft selector");
        for (int p = 0; p < proposal_rows; ++p) {
            const auto selected = result[static_cast<std::size_t>(p)];
            require(selected >= 0 && selected < kVocab,
                    "E5A2 draft proposal outside vocabulary");
        }
    } else {
        for (int p = 0; p < proposal_rows; ++p) {
            dflash_selector_edges_kernel<<<kTopK, 256, 256 * sizeof(float), stream>>>(
                p, anchor, device_seed?m.s.ids:nullptr, m.s.cand_ids,
                m.s.hidden_proj_out, m.pred_cb, m.succ_cb,
                m.s.cand_unary, m.s.edge_scores);
            cuda_check(cudaGetLastError(), "E5A2 launch draft selector edges");
            if (m.position_confidence) {
                dflash_selector_argmax_confidence_kernel<<<1, 1, 0, stream>>>(
                    p, m.s.edge_scores, m.s.cand_ids, m.s.cand_unary,
                    m.s.proposal_out, m.s.confidence_out);
            } else {
                dflash_selector_argmax_kernel<<<1, 1, 0, stream>>>(
                    p, m.s.edge_scores, m.s.cand_ids, m.s.proposal_out);
            }
            cuda_check(cudaGetLastError(), "E5A2 launch draft selector argmax");
            std::int64_t selected = 0;
            cuda_check(cudaMemcpyAsync(&selected, m.s.proposal_out + p, sizeof(selected),
                                       cudaMemcpyDeviceToHost, stream),
                       "E5A2 read draft proposal");
            if (m.position_confidence)
                cuda_check(cudaMemcpyAsync(
                               &m.last_position_confidence[static_cast<std::size_t>(p)],
                               m.s.confidence_out + p, sizeof(DraftPositionConfidence),
                               cudaMemcpyDeviceToHost, stream),
                           "E5A2 read draft position confidence");
            if (p + 1 == proposal_rows)
                timing_close(Impl::TimingCategory::Selector, "selector_serial_final_sync");
            cuda_check(cudaStreamSynchronize(stream), "E5A2 synchronize draft selector");
            require(selected >= 0 && selected < kVocab,
                    "E5A2 draft proposal outside vocabulary");
            result[static_cast<std::size_t>(p)] = selected;
            anchor = selected;
        }
    }
    if(device_seed) {
        std::int64_t invalid=1;
        cuda_check(cudaMemcpyAsync(&invalid,m.s.proposal_out+7,sizeof(invalid),
            cudaMemcpyDeviceToHost,stream),"E5A3 read device seed validity");
        cuda_check(cudaStreamSynchronize(stream),"E5A3 synchronize device seed validity");
        require(!invalid,"E5A3 authoritative device seed invalid/nonfinite");
        require(m.ring_revision==seed_ring_revision &&
                    host_ring_lineage(seed_ring_parent,device_seed->acquisition,
                        device_seed->execution),
                "E5A3 device seed ring witness changed during proposal");
    }
    ph_end("selector");
    if (projection_timing.enabled) {
        const auto before_resolution = std::chrono::steady_clock::now();
        const double api_before_resolution_ms =
            std::chrono::duration<double, std::milli>(before_resolution - projection_api_begin).count();
        float gpu_total_ms_f = 0.0F;
        cuda_check(cudaEventElapsedTime(&gpu_total_ms_f, projection_timing.events[0],
                                        projection_timing.events[projection_timing.marker_count - 1]),
                   "P2 projection timing total elapsed");
        double generic_ms = 0.0, candidate_ms = 0.0, shape4_ms = 0.0;
        double other_projection_ms = 0.0;
        double dense_misc_ms = 0.0, attention_ms = 0.0, h6_ms = 0.0, selector_ms = 0.0;
        int layer_projection_calls = 0, generic_calls = 0, candidate_calls = 0;
        int shape4_calls = 0;
        int other_projection_calls = 0;
        std::vector<float> segment_ms;
        segment_ms.reserve(projection_timing.segments.size());
        for (const auto& segment : projection_timing.segments) {
            float elapsed_ms = 0.0F;
            cuda_check(cudaEventElapsedTime(&elapsed_ms,
                                            projection_timing.events[segment.begin_event],
                                            projection_timing.events[segment.end_event]),
                       "P2 projection timing segment elapsed");
            segment_ms.push_back(elapsed_ms);
            switch (segment.category) {
                case Impl::TimingCategory::Projection:
                    ++layer_projection_calls;
                    if (segment.dispatch == "generic_tile") {
                        ++generic_calls;
                        generic_ms += elapsed_ms;
                    } else if (segment.dispatch == "draft_small_m_mma_split") {
                        ++candidate_calls;
                        candidate_ms += elapsed_ms;
                    } else if (segment.dispatch == "m1_shape4_split") {
                        ++shape4_calls;
                        shape4_ms += elapsed_ms;
                    } else {
                        ++other_projection_calls;
                        other_projection_ms += elapsed_ms;
                    }
                    break;
                case Impl::TimingCategory::OtherProjection:
                    ++other_projection_calls;
                    other_projection_ms += elapsed_ms;
                    break;
                case Impl::TimingCategory::DenseMisc: dense_misc_ms += elapsed_ms; break;
                case Impl::TimingCategory::Attention: attention_ms += elapsed_ms; break;
                case Impl::TimingCategory::H6: h6_ms += elapsed_ms; break;
                case Impl::TimingCategory::Selector: selector_ms += elapsed_ms; break;
            }
        }
        const auto after_resolution = std::chrono::steady_clock::now();
        const double resolution_cpu_ms =
            std::chrono::duration<double, std::milli>(after_resolution - before_resolution).count();
        const double api_wall_ms =
            std::chrono::duration<double, std::milli>(after_resolution - projection_api_begin).count();
        const double gpu_total_ms = gpu_total_ms_f;
        const double attributed_ms = generic_ms + candidate_ms + shape4_ms + other_projection_ms +
                                     dense_misc_ms + attention_ms + h6_ms + selector_ms;
        const double unattributed_ms = gpu_total_ms - attributed_ms;
        const double attributed_pct = gpu_total_ms > 0.0 ? 100.0 * attributed_ms / gpu_total_ms : 0.0;

        std::cout << "P2_PROJECTION_TIMING_CALL_HEADER,proposal,route,role,layer,name,M,K,in,out,dispatch,gpu_ms\n";
        for (std::size_t i = 0; i < projection_timing.segments.size(); ++i) {
            const auto& segment = projection_timing.segments[i];
            if (segment.category != Impl::TimingCategory::Projection &&
                segment.category != Impl::TimingCategory::OtherProjection &&
                segment.category != Impl::TimingCategory::H6) continue;
            const char* role = segment.category == Impl::TimingCategory::Projection
                ? "draft_layer" : (segment.category == Impl::TimingCategory::H6
                    ? "target_h6" : "context_or_fc");
            std::cout << "P2_PROJECTION_TIMING_CALL," << projection_timing.proposal_sequence << ','
                      << (use_ring ? "ring" : "window") << ',' << role << ','
                      << segment.layer << ',' << segment.scope << ',' << segment.rows << ','
                      << segment.metadata.K << ',' << segment.metadata.in_features << ','
                      << segment.metadata.out_features << ',' << segment.dispatch << ','
                      << std::setprecision(9) << segment_ms[i] << '\n';
        }
        std::cout << "P2_PROJECTION_TIMING_MISC_HEADER,proposal,route,scope,gpu_ms\n";
        for (std::size_t i = 0; i < projection_timing.segments.size(); ++i) {
            const auto& segment = projection_timing.segments[i];
            if (segment.category != Impl::TimingCategory::DenseMisc) continue;
            std::cout << "P2_PROJECTION_TIMING_MISC," << projection_timing.proposal_sequence
                      << ',' << (use_ring ? "ring" : "window") << ',' << segment.scope
                      << ',' << std::setprecision(9) << segment_ms[i] << '\n';
        }
        std::cout << "P2_PROJECTION_TIMING_SUMMARY,proposal=" << projection_timing.proposal_sequence
                  << ",route=" << (use_ring ? "ring" : "window")
                  << ",ring_window=" << eff_window
                  << ",B=" << block_len
                  << ",proposal_rows=" << proposal_rows
                  << ",segments=" << projection_timing.segments.size()
                  << ",layer_projection_calls=" << layer_projection_calls
                  << ",expected_layer_projection_calls=35"
                  << ",generic_calls=" << generic_calls
                  << ",candidate_calls=" << candidate_calls
                  << ",shape4_calls=" << shape4_calls
                  << ",other_projection_calls=" << other_projection_calls
                  << ",generic_ms=" << generic_ms
                  << ",candidate_ms=" << candidate_ms
                  << ",shape4_ms=" << shape4_ms
                  << ",other_projection_ms=" << other_projection_ms
                  << ",dense_dynconv_norm_residual_ms=" << dense_misc_ms
                  << ",attention_ms=" << attention_ms
                  << ",h6_ms=" << h6_ms
                  << ",selector_finalsync_ms=" << selector_ms
                  << ",gpu_total_ms=" << gpu_total_ms
                  << ",attributed_ms=" << attributed_ms
                  << ",attributed_pct=" << attributed_pct
                  << ",unattributed_ms=" << unattributed_ms
                  << ",api_before_resolution_ms=" << api_before_resolution_ms
                  << ",api_wall_ms=" << api_wall_ms
                  << ",api_minus_gpu_ms=" << (api_before_resolution_ms - gpu_total_ms)
                  << ",event_record_cpu_ms=" << projection_timing.record_cpu_ms
                  << ",event_resolution_cpu_ms=" << resolution_cpu_ms
                  << ",control=external_uninstrumented" << std::defaultfloat << '\n';
        if (use_ring) {
            const bool coherent_k5_dispatch =
                (generic_calls == 25 && candidate_calls == 0) ||
                (generic_calls == 0 && candidate_calls == 25);
            require(layer_projection_calls == 35 && coherent_k5_dispatch && shape4_calls == 10 &&
                        other_projection_calls == 0,
                    "P2 cached projection timing did not observe coherent 25 K5 + 10 shape4 calls");
        }
    }
    if (e5a3_timing) {
        float tot_ms = 0.0f;
        cuda_check(cudaEventRecord(tot_b, stream), "E5A3 timing total end");
        cuda_check(cudaEventSynchronize(tot_b), "E5A3 timing total sync");
        cuda_check(cudaEventElapsedTime(&tot_ms, tot_a, tot_b), "E5A3 timing total elapsed");
        std::cout << "E5A3 propose_timing use_ring=" << (use_ring ? 1 : 0)
                  << " total_ms=" << tot_ms
                  << " fast_device_liveness=" << (m.fast_device_liveness ? 1 : 0)
                  << " fused_topk_liveness_calls=" << m.fused_topk_liveness_calls
                  << " fused_selector_calls=" << m.fused_selector_calls
                  << " device_liveness_checks=" << m.device_liveness_checks << "\n";
        for (const auto& kv : ph_ms)
            std::cout << "E5A3 phase " << kv.first << " ms=" << kv.second << "\n";
        cuda_check(cudaEventDestroy(tot_a), "E5A3 timing total destroy a");
        cuda_check(cudaEventDestroy(tot_b), "E5A3 timing total destroy b");
    }
    (void)mask_token_id;  // mask tokens are embedded target-side by block_ids
    return result;
}
struct Exl3Dflash2TapHistory::Impl {
    int capacity = 0;
    std::array<std::unique_ptr<DeviceAllocation>, kLayers> rows{};
    int stored = 0;
    int start_pos = 0;

    void drop_oldest(int drop, cudaStream_t stream) {
        if (drop <= 0) return;
        require(drop <= stored, "E5A2 tap history drop exceeds stored rows");
        const int keep = stored - drop;
        for (int t = 0; t < kLayers; ++t) {
            if (keep > 0) {
                const auto* src = static_cast<const std::uint16_t*>(rows[t]->ptr) +
                    static_cast<std::size_t>(drop) * kHidden;
                auto* dst = static_cast<std::uint16_t*>(rows[t]->ptr);
                dflash_row_shift_kernel<<<(keep * kHidden + 255) / 256, 256, 0, stream>>>(
                    src, dst, kHidden, keep);
                cuda_check(cudaGetLastError(), "E5A2 shift tap history");
            }
        }
        start_pos += drop;
        stored = keep;
    }

    void append(const Exl3TextContext& ctx, int first_ctx_row, int count,
                int abs_begin, cudaStream_t stream) {
        require(count >= 1, "E5A2 tap history append count must be positive");
        if (stored == 0) {
            start_pos = abs_begin;
        } else {
            require(abs_begin == start_pos + stored,
                    "E5A2 tap history capture is not contiguous");
        }
        int count_used = count;
        int first_used = first_ctx_row;
        int abs_used = abs_begin;
        if (count_used > capacity) {
            const int front = count_used - capacity;
            first_used += front;
            abs_used += front;
            count_used = capacity;
        }
        const int overflow = stored + count_used - capacity;
        if (overflow > 0) drop_oldest(overflow, stream);
        require(stored + count_used <= capacity, "E5A2 tap history overflow");
        for (int t = 0; t < kLayers; ++t) {
            ctx.copy_tap_rows_to_device(
                kTapLayers[t], first_used,
                static_cast<std::uint16_t*>(rows[t]->ptr) +
                    static_cast<std::size_t>(stored) * kHidden,
                count_used, stream);
        }
        stored += count_used;
    }
};

Exl3Dflash2TapHistory::Exl3Dflash2TapHistory(int capacity)
    : impl_(std::make_unique<Impl>()) {
    require(capacity >= 1, "E5A2 tap history capacity must be positive");
    capacity_ = capacity;
    impl_->capacity = capacity;
    for (int t = 0; t < kLayers; ++t) {
        impl_->rows[t] = std::make_unique<DeviceAllocation>(
            static_cast<std::size_t>(capacity) * kHidden * sizeof(std::uint16_t),
            "E5A2 tap history rows");
    }
}

Exl3Dflash2TapHistory::~Exl3Dflash2TapHistory() = default;

void Exl3Dflash2TapHistory::reset() {
    impl_->stored = 0;
    impl_->start_pos = 0;
    total_rows_ = 0;
}

void Exl3Dflash2TapHistory::capture_prefill(const Exl3TextContext& ctx, int rows,
                                            cudaStream_t stream) {
    require(rows >= 1, "E5A2 prefill capture count must be positive");
    const int ctx_rows = ctx.captured_tap_rows();
    require(ctx_rows >= rows,
            "E5A2 prefill capture exceeds captured target tap rows");
    const int abs_begin = ctx.position() - rows;
    impl_->append(ctx, 0, rows, abs_begin, stream);
    total_rows_ = impl_->stored;
}

void Exl3Dflash2TapHistory::capture_decode(const Exl3TextContext& ctx,
                                           cudaStream_t stream) {
    require(ctx.position() >= 1, "E5A2 decode capture before any target row");
    impl_->append(ctx, 0, 1, ctx.position() - 1, stream);
    total_rows_ = impl_->stored;
}

int Exl3Dflash2TapHistory::copy_window(const Exl3TextContext& ctx, int window_rows,
                                       std::uint16_t* const* dst,
                                       cudaStream_t stream) const {
    (void)ctx;
    require(window_rows >= 1 && impl_->stored >= window_rows + 1,
            "E5A2 tap history needs at least one row newer than the window");
    require(window_rows <= impl_->capacity,
            "E5A2 tap history window exceeds capacity");
    const int first = impl_->stored - 1 - window_rows;
    for (int t = 0; t < kLayers; ++t) {
        cuda_check(cudaMemcpyAsync(
                       dst[t],
                       static_cast<const std::uint16_t*>(impl_->rows[t]->ptr) +
                           static_cast<std::size_t>(first) * kHidden,
                       static_cast<std::size_t>(window_rows) * kHidden *
                           sizeof(std::uint16_t),
                       cudaMemcpyDeviceToDevice, stream),
                   "E5A2 tap history copy window");
    }
    return impl_->start_pos + first;
}

const std::uint16_t* Exl3Dflash2TapHistory::latest_row_device(
    int layer_index, cudaStream_t) const {
    require(layer_index >= 0 && layer_index < kLayers && impl_->stored > 0,
            "E5A2 tap history latest row unavailable");
    return static_cast<const std::uint16_t*>(impl_->rows[layer_index]->ptr) +
        static_cast<std::size_t>(impl_->stored - 1) * kHidden;
}

} // namespace ninfer::exl3
