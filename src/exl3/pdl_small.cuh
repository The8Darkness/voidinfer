#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ninfer::exl3 {

// Programmatic dependent launch for small decode kernels. The prologue waits
// for the preceding grid (full completion and memory flush) before any read,
// then lets the next PDL-aware grid start its launch. Without the launch
// attribute both instructions are no-ops, so plain launches are unchanged.
#define EXL3_PDL_SMALL_PROLOGUE()                                         \
    do {                                                                  \
        asm volatile("griddepcontrol.wait;" ::: "memory");               \
        asm volatile("griddepcontrol.launch_dependents;");               \
    } while (0)

// NINFER_EXL3_PDL_SMALL=0 launches these kernels without the attribute.
inline bool exl3_pdl_small_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_EXL3_PDL_SMALL");
        const char* pdl = std::getenv("NINFER_EXL3_PDL");
        if (pdl && std::strcmp(pdl, "0") == 0) return false;
        if (!value || std::strcmp(value, "1") == 0) return true;
        if (std::strcmp(value, "0") == 0) return false;
        throw std::invalid_argument("NINFER_EXL3_PDL_SMALL must be 0 or 1");
    }();
    return enabled;
}

namespace pdl_small_detail {
template <std::size_t I, class T, class Tuple>
T pick(Tuple& provided) {
    if constexpr (I < std::tuple_size_v<Tuple>) return static_cast<T>(std::get<I>(provided));
    else return T{};  // trailing kernel defaults in these files are null/zero
}
} // namespace pdl_small_detail

// Launches `kernel` with the given arguments; omitted trailing parameters
// are value-initialized (matching their nullptr/0 defaults).
template <class... KernelArgs, class... CallArgs>
void exl3_launch_small(void (*kernel)(KernelArgs...), dim3 grid, dim3 block,
                       std::size_t shared, cudaStream_t stream, CallArgs&&... call) {
    static_assert(sizeof...(CallArgs) <= sizeof...(KernelArgs), "too many kernel arguments");
    auto provided = std::forward_as_tuple(std::forward<CallArgs>(call)...);
    auto build = [&]<std::size_t... I>(std::index_sequence<I...>) {
        return std::tuple<std::decay_t<KernelArgs>...>{
            pdl_small_detail::pick<I, std::decay_t<KernelArgs>>(provided)...};
    };
    auto values = build(std::index_sequence_for<KernelArgs...>{});
    void* pointers[sizeof...(KernelArgs) ? sizeof...(KernelArgs) : 1];
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        ((pointers[I] = static_cast<void*>(&std::get<I>(values))), ...);
    }(std::index_sequence_for<KernelArgs...>{});
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = shared;
    config.stream = stream;
    const bool pdl = exl3_pdl_small_enabled() && stream != nullptr;
    config.attrs = pdl ? &attribute : nullptr;
    config.numAttrs = pdl ? 1 : 0;
    const cudaError_t error =
        cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(kernel), pointers);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("small PDL launch: ") + cudaGetErrorString(error));
}

} // namespace ninfer::exl3
