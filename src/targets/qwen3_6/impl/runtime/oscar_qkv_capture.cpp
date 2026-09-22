#include "targets/qwen3_6/impl/runtime/oscar_qkv_capture.h"

#include "core/device.h"
#include "targets/qwen3_6/impl/frontend/digest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace ninfer::targets::qwen3_6::oscar_internal {
namespace {

using Json = nlohmann::ordered_json;
using ninfer::targets::qwen3_6::frontend_internal::sha256;
using ninfer::targets::qwen3_6::frontend_internal::sha256_hex;

constexpr std::int32_t kTotalLayers = 64;
constexpr std::int32_t kQueryHeads  = 24;
constexpr std::int32_t kKvHeads     = 4;
constexpr std::int32_t kHeadDim     = 256;
constexpr std::int32_t kFullLayers  = 16;
constexpr std::uint32_t kFullMask    = (1U << kFullLayers) - 1U;
constexpr std::string_view kDType    = "bf16";
constexpr std::string_view kStage =
    "post_qk_rmsnorm_post_rope_pre_causal_attention_cache_append";

[[nodiscard]] const char* required_environment(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        throw std::invalid_argument(std::string("OSCAR QKV capture requires ") + name);
    }
    return value;
}

[[nodiscard]] std::int32_t positive_environment(const char* name, std::int32_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') { return fallback; }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 ||
        parsed > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(std::string("OSCAR QKV capture has invalid ") + name);
    }
    return static_cast<std::int32_t>(parsed);
}

void require_sha256(std::string_view value, const char* label) {
    if (value.size() != 64 ||
        !std::all_of(value.begin(), value.end(), [](unsigned char c) { return std::isxdigit(c); })) {
        throw std::invalid_argument(std::string("OSCAR QKV capture has invalid ") + label);
    }
}

[[nodiscard]] std::string chunk_name(std::int32_t chunk_id) {
    std::ostringstream name;
    name << "chunk_" << std::setw(4) << std::setfill('0') << chunk_id;
    return name.str();
}

[[nodiscard]] std::filesystem::path write_tensor(const std::filesystem::path& root,
                                                  std::int32_t model_layer, std::int32_t chunk_id,
                                                  std::string_view type,
                                                  const std::vector<std::uint8_t>& bytes) {
    if (bytes.empty()) { throw std::runtime_error("OSCAR QKV capture produced an empty tensor"); }
    const std::filesystem::path relative =
        std::filesystem::path("layer_" + std::to_string(model_layer)) / chunk_name(chunk_id) /
        (std::string(type) + ".bin");
    const std::filesystem::path path = root / relative;
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) { throw std::runtime_error("OSCAR QKV capture cannot open " + path.string()); }
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    output.flush();
    if (!output) { throw std::runtime_error("OSCAR QKV capture cannot write " + path.string()); }
    return relative;
}

[[nodiscard]] std::vector<std::uint8_t> copy_tensor(const Tensor& tensor, std::int32_t heads,
                                                    std::int32_t tokens, const char* label,
                                                    cudaStream_t stream) {
    if (tensor.dtype != DType::BF16 || tensor.data == nullptr || !tensor.is_contiguous() ||
        tensor.ne[0] != kHeadDim || tensor.ne[1] != heads || tensor.ne[2] != tokens ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("OSCAR QKV capture shape/dtype mismatch for ") +
                                    label);
    }
    const std::size_t expected_bytes =
        static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(heads) *
        static_cast<std::size_t>(tokens) * sizeof(std::uint16_t);
    if (tensor.bytes() != expected_bytes) {
        throw std::invalid_argument(std::string("OSCAR QKV capture byte size mismatch for ") +
                                    label);
    }
    std::vector<std::uint8_t> host(expected_bytes);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), tensor.data, expected_bytes, cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::size_t offset = 0; offset < host.size(); offset += sizeof(std::uint16_t)) {
        const std::uint16_t bits = static_cast<std::uint16_t>(host[offset]) |
                                   (static_cast<std::uint16_t>(host[offset + 1]) << 8U);
        if ((bits & 0x7f80U) == 0x7f80U) {
            throw std::runtime_error(std::string("OSCAR QKV capture found NaN/Inf in ") + label);
        }
    }
    return host;
}

[[nodiscard]] OscarQKVTensorRecord make_tensor_record(const std::filesystem::path& root,
                                                       std::int32_t model_layer,
                                                       std::int32_t chunk_id,
                                                       std::string_view type,
                                                       std::vector<std::int32_t> dimensions,
                                                       const std::vector<std::uint8_t>& bytes) {
    const std::filesystem::path relative =
        write_tensor(root, model_layer, chunk_id, type, bytes);
    return OscarQKVTensorRecord{
        .type       = std::string(type),
        .file       = relative.generic_string(),
        .sha256     = sha256_hex(sha256(std::span<const std::uint8_t>(bytes.data(), bytes.size()))),
        .dimensions = std::move(dimensions),
        .bytes      = bytes.size(),
    };
}

} // namespace

OscarQKVCapture::OscarQKVCapture()
    : root_(std::filesystem::absolute(required_environment("NINFER_OSCAR_QKV_CAPTURE_DIR"))
                .lexically_normal()),
      model_id_(std::getenv("NINFER_OSCAR_QKV_MODEL_ID") != nullptr
                    ? std::getenv("NINFER_OSCAR_QKV_MODEL_ID")
                    : "qwen3.8-27b"),
      weights_id_(std::getenv("NINFER_OSCAR_QKV_WEIGHTS_ID") != nullptr
                      ? std::getenv("NINFER_OSCAR_QKV_WEIGHTS_ID")
                      : "nvfp4-dflash2"),
      model_sha256_(required_environment("NINFER_OSCAR_QKV_MODEL_SHA256")),
      executable_sha256_(required_environment("NINFER_OSCAR_QKV_EXECUTABLE_SHA256")),
      source_identity_(required_environment("NINFER_OSCAR_QKV_SOURCE_ID")),
      input_description_(std::getenv("NINFER_OSCAR_QKV_INPUT_DESCRIPTION") != nullptr
                             ? std::getenv("NINFER_OSCAR_QKV_INPUT_DESCRIPTION")
                             : "deterministic_token_prompt"),
      expected_tokens_(positive_environment("NINFER_OSCAR_QKV_EXPECTED_TOKENS", 256)) {
    require_sha256(model_sha256_, "model SHA-256");
    require_sha256(executable_sha256_, "executable SHA-256");
    if (source_identity_.empty()) {
        throw std::invalid_argument("OSCAR QKV capture source identity is empty");
    }
    if (expected_tokens_ < 1 || expected_tokens_ > 1'048'576) {
        throw std::invalid_argument("OSCAR QKV capture expected token count is out of range");
    }
    if (input_description_.empty()) {
        throw std::invalid_argument("OSCAR QKV capture input description is empty");
    }
    std::filesystem::create_directories(root_);
    if (!std::filesystem::is_directory(root_)) {
        throw std::invalid_argument("OSCAR QKV capture root is not a directory");
    }
    if (std::filesystem::exists(root_ / "manifest.json")) {
        throw std::invalid_argument("OSCAR QKV capture manifest already exists; use a fresh root");
    }
    if (std::filesystem::directory_iterator(root_) != std::filesystem::directory_iterator{}) {
        throw std::invalid_argument("OSCAR QKV capture root must be empty");
    }
}

OscarQKVCapture::~OscarQKVCapture() {
    if (!finalized_) {
        try {
            write_manifest();
        } catch (const std::exception& error) {
            std::cerr << "OSCAR QKV capture finalization failed: " << error.what() << '\n';
        }
    }
}

void OscarQKVCapture::capture(std::int32_t model_layer, std::int32_t full_attention_index,
                              const Tensor& q, const Tensor& k, const Tensor& v,
                              cudaStream_t stream) {
    if (finalized_) { throw std::logic_error("OSCAR QKV capture was already finalized"); }
    if (model_layer != 3 + 4 * full_attention_index || full_attention_index < 0 ||
        full_attention_index >= kFullLayers) {
        throw std::invalid_argument("OSCAR QKV capture received a non-full-attention layer");
    }
    if (current_chunk_mask_ & (1U << full_attention_index)) {
        throw std::logic_error("OSCAR QKV capture received a duplicate full-attention layer");
    }
    if (full_attention_index != static_cast<std::int32_t>(std::popcount(current_chunk_mask_))) {
        throw std::logic_error("OSCAR QKV capture full-attention layer order changed");
    }
    const std::int32_t tokens = q.ne[2];
    if (tokens <= 0 || tokens > expected_tokens_ ||
        completed_tokens_ + std::max(current_chunk_tokens_, tokens) > expected_tokens_) {
        throw std::invalid_argument("OSCAR QKV capture token count exceeds the smoke budget");
    }
    if (current_chunk_tokens_ == 0) {
        current_chunk_tokens_ = tokens;
    } else if (current_chunk_tokens_ != tokens) {
        throw std::logic_error("OSCAR QKV capture Q/K/V chunk token count changed");
    }
    if (cudaStreamCaptureStatus status{}; cudaStreamIsCapturing(stream, &status) == cudaSuccess &&
                                              status != cudaStreamCaptureStatusNone) {
        throw std::logic_error("OSCAR QKV capture cannot run during CUDA graph capture");
    }

    const std::vector<std::uint8_t> q_host = copy_tensor(q, kQueryHeads, tokens, "Q", stream);
    const std::vector<std::uint8_t> k_host = copy_tensor(k, kKvHeads, tokens, "K", stream);
    const std::vector<std::uint8_t> v_host = copy_tensor(v, kKvHeads, tokens, "V", stream);
    OscarQKVCaptureRecord record{
        .model_layer          = model_layer,
        .full_attention_index = full_attention_index,
        .tokens               = tokens,
        .chunk_id             = current_chunk_id_,
        .tensors              = {
            make_tensor_record(root_, model_layer, current_chunk_id_, "q",
                               {tokens, kQueryHeads, kHeadDim}, q_host),
            make_tensor_record(root_, model_layer, current_chunk_id_, "k",
                               {tokens, kKvHeads, kHeadDim}, k_host),
            make_tensor_record(root_, model_layer, current_chunk_id_, "v",
                               {tokens, kKvHeads, kHeadDim}, v_host),
        },
    };
    records_.push_back(std::move(record));
    current_chunk_mask_ |= 1U << full_attention_index;
    if (current_chunk_mask_ == kFullMask) {
        completed_tokens_ += current_chunk_tokens_;
        current_chunk_tokens_ = 0;
        current_chunk_mask_   = 0;
        ++current_chunk_id_;
    }
}

void OscarQKVCapture::write_manifest() {
    const bool complete = current_chunk_mask_ == 0 && completed_tokens_ == expected_tokens_ &&
                          !records_.empty();
    Json manifest{
        {"schema", "oscar-qkv-v1"},
        {"post_rope_qkv", true},
        {"dtype", kDType},
        {"capture_stage", kStage},
        {"complete", complete},
        {"useful_tokens", completed_tokens_},
        {"expected_useful_tokens", expected_tokens_},
        {"model",
         {{"id", model_id_},
          {"weights_id", weights_id_},
          {"sha256", model_sha256_},
          {"total_layers", kTotalLayers},
          {"query_heads", kQueryHeads},
          {"kv_heads", kKvHeads},
          {"head_dim", kHeadDim},
          {"rotary_dim", 64}}},
        {"executable", {{"sha256", executable_sha256_}}},
        {"source_identity", source_identity_},
        {"gqa",
         {{"ratio", kQueryHeads / kKvHeads},
          {"q_head_to_kv_head",
           [] {
               Json mapping = Json::array();
               for (std::int32_t head = 0; head < kQueryHeads; ++head) {
                   mapping.push_back(head / (kQueryHeads / kKvHeads));
               }
               return mapping;
           }()}}},
        {"input", {{"kind", input_description_}}},
        {"captures", Json::array()},
        {"tensor_records", Json::array()},
    };

    std::size_t dump_bytes = 0;
    for (const OscarQKVCaptureRecord& capture : records_) {
        Json item{
            {"model_layer", capture.model_layer},
            {"full_attention_index", capture.full_attention_index},
            {"kind", "full_attention"},
            {"dtype", kDType},
            {"tokens", capture.tokens},
            {"q_heads", kQueryHeads},
            {"kv_heads", kKvHeads},
            {"head_dim", kHeadDim},
            {"chunk_id", capture.chunk_id},
            {"capture_stage", kStage},
            {"dimensions",
             {{"q", capture.tensors[0].dimensions},
              {"k", capture.tensors[1].dimensions},
              {"v", capture.tensors[2].dimensions}}},
            {"files",
             {{"q", capture.tensors[0].file},
              {"k", capture.tensors[1].file},
              {"v", capture.tensors[2].file}}},
            {"sha256",
             {{"q", capture.tensors[0].sha256},
              {"k", capture.tensors[1].sha256},
              {"v", capture.tensors[2].sha256}}},
        };
        manifest["captures"].push_back(std::move(item));
        for (const OscarQKVTensorRecord& tensor : capture.tensors) {
            dump_bytes += tensor.bytes;
            manifest["tensor_records"].push_back(
                {{"model_sha256", model_sha256_},
                 {"executable_sha256", executable_sha256_},
                 {"source_identity", source_identity_},
                 {"layer_index", capture.model_layer},
                 {"full_attention_index", capture.full_attention_index},
                 {"qkv_type", tensor.type},
                 {"dtype", kDType},
                 {"dimensions", tensor.dimensions},
                 {"q_heads", kQueryHeads},
                 {"kv_heads", kKvHeads},
                 {"head_dim", kHeadDim},
                 {"token_count", capture.tokens},
                 {"chunk_id", capture.chunk_id},
                 {"capture_stage", kStage},
                 {"file", tensor.file},
                 {"byte_count", tensor.bytes},
                 {"data_sha256", tensor.sha256}});
        }
    }
    manifest["dump_bytes"] = dump_bytes;
    manifest["capture_count"] = records_.size();
    const std::filesystem::path temporary = root_ / "manifest.json.tmp";
    const std::filesystem::path manifest_path = root_ / "manifest.json";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) { throw std::runtime_error("OSCAR QKV capture cannot open manifest"); }
    output << manifest.dump(2) << '\n';
    output.flush();
    if (!output) { throw std::runtime_error("OSCAR QKV capture cannot write manifest"); }
    output.close();
    if (!output) { throw std::runtime_error("OSCAR QKV capture cannot close manifest"); }
    std::filesystem::rename(temporary, manifest_path);
}

void OscarQKVCapture::finalize() {
    if (finalized_) { return; }
    write_manifest();
    finalized_ = true;
    if (current_chunk_mask_ != 0 || completed_tokens_ != expected_tokens_ || records_.empty()) {
        throw std::runtime_error("OSCAR QKV capture is incomplete");
    }
}

OscarQKVCapture* qkv_capture_from_environment() {
    static std::unique_ptr<OscarQKVCapture> capture = [] {
        const char* root = std::getenv("NINFER_OSCAR_QKV_CAPTURE_DIR");
        if (root == nullptr || *root == '\0') { return std::unique_ptr<OscarQKVCapture>{}; }
        return std::make_unique<OscarQKVCapture>();
    }();
    return capture.get();
}

void finalize_qkv_capture_from_environment() {
    if (OscarQKVCapture* capture = qkv_capture_from_environment(); capture != nullptr) {
        capture->finalize();
    }
}

} // namespace ninfer::targets::qwen3_6::oscar_internal
