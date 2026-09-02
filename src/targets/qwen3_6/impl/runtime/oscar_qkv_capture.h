#pragma once

#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

namespace ninfer::targets::qwen3_6::oscar_internal {

struct OscarQKVTensorRecord {
    std::string type;
    std::string file;
    std::string sha256;
    std::vector<std::int32_t> dimensions;
    std::size_t bytes = 0;
};

struct OscarQKVCaptureRecord {
    std::int32_t model_layer          = -1;
    std::int32_t full_attention_index = -1;
    std::int32_t tokens               = 0;
    std::int32_t chunk_id             = -1;
    std::array<OscarQKVTensorRecord, 3> tensors;
};

class OscarQKVCapture {
public:
    OscarQKVCapture();
    ~OscarQKVCapture();

    OscarQKVCapture(const OscarQKVCapture&)            = delete;
    OscarQKVCapture& operator=(const OscarQKVCapture&) = delete;

    void capture(std::int32_t model_layer, std::int32_t full_attention_index,
                 const Tensor& q, const Tensor& k, const Tensor& v, cudaStream_t stream);
    void finalize();

private:
    void write_manifest();

    std::filesystem::path root_;
    std::string model_id_;
    std::string weights_id_;
    std::string model_sha256_;
    std::string executable_sha256_;
    std::string source_identity_;
    std::string input_description_;
    std::int32_t expected_tokens_ = 256;
    std::int32_t completed_tokens_ = 0;
    std::int32_t current_chunk_tokens_ = 0;
    std::int32_t current_chunk_id_ = 0;
    std::uint32_t current_chunk_mask_ = 0;
    bool finalized_                         = false;
    std::vector<OscarQKVCaptureRecord> records_;
};

OscarQKVCapture* qkv_capture_from_environment();
void finalize_qkv_capture_from_environment();

} // namespace ninfer::targets::qwen3_6::oscar_internal
