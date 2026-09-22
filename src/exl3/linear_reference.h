#pragma once

#include "exl3/safetensors.h"

#include <cstdint>
#include <string_view>
#include <vector>

namespace ninfer::exl3 {

struct Exl3LinearMetadata {
    int in_features = 0;
    int out_features = 0;
    int K = 0;
    bool mcg = false;
    bool mul1 = false;
    bool has_bias = false;
    std::string_view codebook = "mul1";
};

struct Exl3LinearTensors {
    const TensorPayload* trellis = nullptr;
    const TensorPayload* suh = nullptr;
    const TensorPayload* svh = nullptr;
    const TensorPayload* mul1 = nullptr;
};

// Validate the exact E2 module contract before any payload is interpreted.
void validate_linear_tensors(const Exl3LinearTensors& tensors,
                             const Exl3LinearMetadata& metadata);

struct Exl3ReferenceOutput {
    std::vector<std::uint16_t> fp16_bits;
    std::vector<std::uint64_t> shape;
};

// Deliberately slow, deterministic correctness path. It consumes packed EXL3 tiles directly
// and only reconstructs one 16x16 tile at a time on the stack; it never materializes a dense
// matrix or writes any model artifact.
Exl3ReferenceOutput exl3_linear_reference(const TensorPayload& input,
                                          const Exl3LinearTensors& tensors,
                                          const Exl3LinearMetadata& metadata);

} // namespace ninfer::exl3
