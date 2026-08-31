#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace ninfer::ops {

// Re-quantize one complete [padded_capacity, kv_heads] OSCAR slot image on the GPU. The source
// and destination are independent affine OSCAR representations; conversion is used only at the
// StateImage boundary, so normal L0 append/attention never pays this cost.
void cyclic_oscar_requantize_launch(const std::uint8_t* source_k,
                                    const __nv_bfloat16* source_k_metadata,
                                    const std::uint8_t* source_v,
                                    const __nv_bfloat16* source_v_metadata,
                                    std::uint8_t* destination_k,
                                    __nv_bfloat16* destination_k_metadata,
                                    std::uint8_t* destination_v,
                                    __nv_bfloat16* destination_v_metadata, int source_bits,
                                    int destination_bits, int padded_capacity, int kv_heads,
                                    cudaStream_t stream);

} // namespace ninfer::ops
