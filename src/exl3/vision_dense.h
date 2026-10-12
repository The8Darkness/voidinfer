#pragma once
#include <ninfer/targets/qwen3_6/vision_control.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace ninfer::exl3 {

struct Exl3DenseVisionStats {
    std::size_t patches=0,merged_tokens=0;
    std::size_t scratch_device_bytes=0;
    double encode_ms=0;
};

// Qwen3.8 Vision tower in BF16 (the checkpoint's own weights, no quantization):
// patch projection, 27 pre-norm blocks (2-D RoPE, segment-local attention,
// tanh-GELU MLP) and the 2x2 merger (exact GELU), as in the reference model.
// The 0.9 GB of weights stay in pinned host memory and stream to the device
// one block at a time through a double buffer, so the encoder holds no VRAM
// between requests; activations are allocated per call (stream-ordered).
class Exl3DenseVision {
public:
    // `directory` holds `model.visual.*` BF16 tensors (a Transformers
    // safetensors index, e.g. the BF16 checkpoint of the EXL3 target).
    explicit Exl3DenseVision(const std::filesystem::path& directory);
    ~Exl3DenseVision();
    Exl3DenseVision(const Exl3DenseVision&)=delete;
    Exl3DenseVision& operator=(const Exl3DenseVision&)=delete;

    // One prepared item (BF16 patches) -> FP16 rows [merged_count][5120] at
    // `output` (device). Ordered on `stream`; returns after the stream fence.
    Exl3DenseVisionStats encode(const targets::qwen3_6::PreparedMediaPayload& payload,
        const targets::qwen3_6::VisionItemControl& control,std::uint16_t* output,
        cudaStream_t stream);
    // One synthetic 32x32-patch image: loads the tower's kernels and cuBLAS
    // heuristics at startup instead of on the first media request.
    void warm_up(cudaStream_t stream);
    std::size_t host_bytes() const noexcept;
    static constexpr int kOutputWidth=5120;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::exl3
