#pragma once

#include "core/arena.h"
#include "core/oscar_mixed_attention_reference.h"
#include "core/tensor.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

namespace ninfer::targets::qwen3_6::oscar_internal {

enum class RotationPrecision : std::uint8_t {
    Bf16Materialized,
    Fp32Rotation,
    Fp32Inverse,
    Fp32RotationAndInverse,
};

// This is deliberately a runtime-only representation of the immutable C4 FP32 assets.  The
// loader never depends on LibTorch or on the Python .pt serialization format.
class OscarRotationSet {
public:
    OscarRotationSet(const OscarRotationSet&)            = delete;
    OscarRotationSet& operator=(const OscarRotationSet&) = delete;
    OscarRotationSet(OscarRotationSet&&)                 = delete;
    OscarRotationSet& operator=(OscarRotationSet&&)      = delete;
    ~OscarRotationSet();

    [[nodiscard]] const std::string& asset_identity() const noexcept { return asset_identity_; }
    [[nodiscard]] const std::string& asset_hash() const noexcept { return asset_hash_; }
    [[nodiscard]] const std::string& model_sha256() const noexcept { return model_sha256_; }
    [[nodiscard]] std::int32_t full_attention_layer_count() const noexcept { return 16; }
    [[nodiscard]] std::span<const float> k_matrix_host(std::int32_t model_layer) const;
    [[nodiscard]] std::span<const float> v_matrix_host(std::int32_t model_layer) const;

    // Input/output tensors are contiguous BF16 [head_dim, heads, tokens] views.  Q and K use
    // R_K on the right in the row-vector convention; V uses R_V.  The inverse recovers V @
    // R_V^T before the existing gate and output projection.
    void rotate_qkv(std::int32_t model_layer, const Tensor& q, const Tensor& k, const Tensor& v,
                    Tensor& q_rotated, Tensor& k_rotated, Tensor& v_rotated,
                    cudaStream_t stream) const;
    void rotate_qkv_fp32(std::int32_t model_layer, const Tensor& q, const Tensor& k,
                         const Tensor& v, Tensor& q_rotated, Tensor& k_rotated,
                         Tensor& v_rotated, cudaStream_t stream) const;
    void reference_attention_fp32(const Tensor& q, const Tensor& k, const Tensor& v, float scale,
                                  Tensor& out, cudaStream_t stream) const;
    void inverse_value(std::int32_t model_layer, const Tensor& rotated, Tensor& recovered,
                       cudaStream_t stream) const;
    void inverse_value_fp32(std::int32_t model_layer, const Tensor& rotated, Tensor& recovered,
                            cudaStream_t stream) const;
    void fp32_to_bf16(const Tensor& input, Tensor& output, cudaStream_t stream) const;

    OscarRotationSet(std::string asset_identity, std::string asset_hash, std::string model_sha256,
                     DeviceBuffer k_rotation, DeviceBuffer v_rotation,
                     std::vector<float> k_rotation_host, std::vector<float> v_rotation_host);

private:
    [[nodiscard]] const DeviceBuffer& k_rotation() const noexcept { return k_rotation_; }
    [[nodiscard]] const DeviceBuffer& v_rotation() const noexcept { return v_rotation_; }

    std::string asset_identity_;
    std::string asset_hash_;
    std::string model_sha256_;
    DeviceBuffer k_rotation_;
    DeviceBuffer v_rotation_;
    std::vector<float> k_rotation_host_;
    std::vector<float> v_rotation_host_;
};

// Diagnostic-only cache used by D1.2.  It is deliberately separate from PagedKVCache: the
// production cache has no FP32 representation, while this cache must make the unrotated and
// rotated comparisons share exactly the same persistent numerical path across prefill/decode.
class OscarMatchedFP32Cache {
public:
    explicit OscarMatchedFP32Cache(std::uint32_t max_context);
    ~OscarMatchedFP32Cache();

    OscarMatchedFP32Cache(const OscarMatchedFP32Cache&)            = delete;
    OscarMatchedFP32Cache& operator=(const OscarMatchedFP32Cache&) = delete;

    void reset(cudaStream_t stream);
    void append(std::int32_t full_layer, const Tensor& k, const Tensor& v,
                const Tensor& positions, cudaStream_t stream);
    void copy_layer_prefix(std::int32_t full_layer, std::uint32_t token_count, float* k_host,
                           float* v_host, cudaStream_t stream) const;
    void attention(std::int32_t full_layer, const Tensor& q, const Tensor& positions, float scale,
                   Tensor& out, cudaStream_t stream) const;

    [[nodiscard]] std::uint32_t max_context() const noexcept { return max_context_; }

private:
    std::uint32_t max_context_ = 0;
    DeviceBuffer k_cache_;
    DeviceBuffer v_cache_;
};

// Empty/unset mode is the normal runtime.  Any non-empty unsupported mode fails closed.
[[nodiscard]] bool rotated_bf16_mode_enabled();
[[nodiscard]] bool live_int2_reference_mode_enabled();
[[nodiscard]] bool live_int2_gpu_mode_enabled();
[[nodiscard]] bool live_int2_gpu_resident_mode_enabled();
// D4.6 fused resident attention is the default. Set NINFER_OSCAR_D4_6_FUSED=0 only for an
// explicit D4.5 three-stage control measurement.
[[nodiscard]] bool live_int2_gpu_fused_mode_enabled();
// D4.5 prefill query block selected by NINFER_OSCAR_D4_5_QBLOCK; defaults to Q64 and accepts
// only the benchmarked candidates Q8/Q16/Q32/Q64.
[[nodiscard]] std::uint32_t live_gpu_prefill_query_block_size();
[[nodiscard]] bool matched_fp32_mode_enabled();
[[nodiscard]] RotationPrecision rotation_precision_mode();
[[nodiscard]] std::shared_ptr<const OscarRotationSet> rotation_set_from_environment();
[[nodiscard]] std::shared_ptr<OscarMatchedFP32Cache>
matched_fp32_cache_for(const void* owner, std::uint32_t max_context);

// Diagnostic-only live reader. It owns one typed D2.2 cache per verified full-attention layer,
// accepts actual runtime rows already transformed into the calibrated OSCAR coordinate system,
// and invokes the independent, scalar D2.3a reader. No GDN state or production StateImage is
// reachable through this object.
class OscarLiveMixedReferenceCache {
public:
    struct ProfileTotals {
        std::uint64_t append_calls = 0;
        std::uint64_t aging_events = 0;
        std::uint64_t attention_calls = 0;
        double qkv_rotation_us = 0.0;
        double aging_us = 0.0;
        double aging_reference_encode_us = 0.0;
        double aging_parity_check_us = 0.0;
        double reader_q_rotation_us = 0.0;
        double int2_k_decode_us = 0.0;
        double qk_us = 0.0;
        double softmax_us = 0.0;
        double int2_v_decode_us = 0.0;
        double av_us = 0.0;
        double rv_inverse_us = 0.0;
        double reader_total_us = 0.0;
        double full_attention_us = 0.0;
        double gpu_cache_staging_us = 0.0;
        double gpu_mixed_kernel_us = 0.0;
        double gpu_recovery_us = 0.0;
        double gpu_fused_kernel_us = 0.0;
        double gpu_prefill_full_attention_us = 0.0;
        double gpu_decode_full_attention_us = 0.0;
        std::uint64_t gpu_cache_staging_bytes = 0;
        std::uint64_t gpu_cache_staging_calls = 0;
        std::uint64_t gpu_attention_calls = 0;
        std::uint64_t gpu_attention_batches = 0;
        std::uint64_t gpu_attention_kernel_launches = 0;
        std::uint64_t gpu_prefill_queries = 0;
        std::uint64_t gpu_prefill_batches = 0;
        std::uint64_t gpu_decode_queries = 0;
        std::uint64_t gpu_decode_batches = 0;
        double gpu_resident_publish_us = 0.0;
        double gpu_resident_aging_us = 0.0;
        std::uint64_t gpu_resident_append_calls = 0;
        std::uint64_t gpu_resident_aging_events = 0;
        std::uint64_t gpu_resident_codec_parity_checks = 0;
        std::uint64_t gpu_incremental_host_device_bytes = 0;
        std::uint64_t gpu_resident_cache_bytes = 0;
        std::uint64_t gpu_resident_workspace_bytes = 0;
        std::uint64_t gpu_fused_kv_tiles = 0;
        std::uint64_t gpu_fused_query_tiles = 0;
    };

    OscarLiveMixedReferenceCache(std::uint32_t max_context, std::uint64_t sequence_id,
                                 std::shared_ptr<const OscarRotationSet> rotations);
    ~OscarLiveMixedReferenceCache();

    OscarLiveMixedReferenceCache(const OscarLiveMixedReferenceCache&)            = delete;
    OscarLiveMixedReferenceCache& operator=(const OscarLiveMixedReferenceCache&) = delete;

    void reset();
    void append(std::int32_t model_layer, std::uint32_t logical_token,
                std::span<const std::uint16_t> k_bf16,
                std::span<const std::uint16_t> v_bf16);
    [[nodiscard]] ::ninfer::OscarMixedAttentionTrace
    attention(std::int32_t model_layer, std::uint32_t query_token,
              std::span<const float> q_original);
    // First live GPU integration for D4.3. The mixed cache remains the qualified typed host
    // representation; prepare_gpu() stages its three typed regions into device views once for
    // the current layer invocation, and attention_gpu() dispatches the D4.2b kernel directly.
    // No CPU attention is used by this path. The caller owns the FP32 Q/output tensors.
    void prepare_gpu(std::int32_t model_layer, cudaStream_t stream);
    // D4.4 path: publish actual rotated FP32 runtime rows directly into persistent device
    // prefix/recent storage and packed historical storage.  The host typed cache is populated
    // only when an explicit reference-validation environment flag is enabled.
    void append_gpu(std::int32_t model_layer, std::uint32_t logical_start,
                    const Tensor& rotated_k, const Tensor& rotated_v, cudaStream_t stream);
    void attention_gpu_batch(std::int32_t model_layer, std::uint32_t query_start,
                             const Tensor& q_rotated, const Tensor& rotated_output,
                             std::uint32_t query_count, cudaStream_t stream);
    void attention_gpu(std::int32_t model_layer, std::uint32_t query_token,
                       const Tensor& q_rotated, const Tensor& rotated_output,
                       cudaStream_t stream);
    [[nodiscard]] const ::ninfer::OscarMixedAgingLayerCache& layer(
        std::int32_t model_layer) const;
    [[nodiscard]] ::ninfer::OscarMixedCacheAccounting accounting() const;
    void record_qkv_rotation_us(double value) noexcept;
    void record_full_attention_us(double value) noexcept;
    void record_gpu_mixed_kernel_us(double value) noexcept;
    void record_gpu_recovery_us(double value) noexcept;
    void record_gpu_fused_kernel_us(double value) noexcept;
    void record_gpu_phase_full_attention_us(bool prefill, double value) noexcept;
    void record_gpu_incremental_host_device_bytes(std::uint64_t value) noexcept;
    void record_gpu_prefill_batch(std::uint32_t query_count) noexcept;
    void record_gpu_decode_batch(std::uint32_t query_count) noexcept;
    void refresh_live_reference_taps();
    [[nodiscard]] const ProfileTotals& profile_totals() const noexcept { return profile_; }
    [[nodiscard]] bool gpu_resident_mode_enabled() const noexcept {
        return gpu_resident_enabled_;
    }
    [[nodiscard]] const std::shared_ptr<const OscarRotationSet>& rotations() const noexcept {
        return rotations_;
    }
    [[nodiscard]] std::uint32_t max_context() const noexcept { return max_context_; }

private:
    std::uint32_t max_context_ = 0;
    std::uint64_t sequence_id_ = 0;
    std::shared_ptr<const OscarRotationSet> rotations_;
    ::ninfer::OscarMixedAgingAssetContract asset_;
    std::array<std::unique_ptr<::ninfer::OscarMixedAgingLayerCache>, 16> layers_{};
    std::array<bool, 16> dispatch_seen_{};
    std::uint32_t aging_parity_checks_ = 0;
    ProfileTotals profile_{};
    std::filesystem::path tap_root_;
    std::vector<std::uint32_t> tap_layers_;
    std::vector<std::uint32_t> tap_queries_;

    struct GpuResidentLayer {
        DeviceBuffer prefix_k;
        DeviceBuffer prefix_v;
        DeviceBuffer historical_k;
        DeviceBuffer historical_v;
        DeviceBuffer historical_k_metadata;
        DeviceBuffer historical_v_metadata;
        DeviceBuffer recent_k;
        DeviceBuffer recent_v;
        std::uint32_t context = 0;
        std::uint32_t recent_head = 0;
    };

    bool gpu_resident_enabled_ = false;
    bool gpu_fused_enabled_ = false;
    bool gpu_host_oracle_enabled_ = false;
    std::array<GpuResidentLayer, 16> gpu_resident_layers_{};
    DeviceBuffer gpu_resident_scores_;
    DeviceBuffer gpu_resident_softmax_;
    DeviceBuffer gpu_resident_fused_decode_workspace_;
    std::uint64_t gpu_resident_cache_bytes_ = 0;
    std::uint64_t gpu_resident_workspace_bytes_ = 0;
    std::uint32_t gpu_prefill_query_block_size_ = 64;

    DeviceBuffer gpu_prefix_k_;
    DeviceBuffer gpu_prefix_v_;
    DeviceBuffer gpu_historical_k_;
    DeviceBuffer gpu_historical_v_;
    DeviceBuffer gpu_historical_k_metadata_;
    DeviceBuffer gpu_historical_v_metadata_;
    DeviceBuffer gpu_recent_k_;
    DeviceBuffer gpu_recent_v_;
    DeviceBuffer gpu_scores_;
    DeviceBuffer gpu_softmax_;
    std::int32_t gpu_staged_layer_ = -1;
    std::uint32_t gpu_staged_context_ = 0;
    std::uint32_t gpu_staged_prefix_ = 0;
    std::uint32_t gpu_staged_historical_ = 0;
    std::uint32_t gpu_staged_recent_ = 0;

    void initialize_layers();
    void initialize_gpu_resident_storage();
    void write_tap_if_selected(std::int32_t model_layer, std::uint32_t query_token,
                               std::span<const float> q_original,
                               const ::ninfer::OscarMixedAttentionTrace& trace) const;
};

[[nodiscard]] std::shared_ptr<OscarLiveMixedReferenceCache>
live_mixed_cache_for(const void* owner, std::uint32_t max_context,
                     std::shared_ptr<const OscarRotationSet> rotations);

// Diagnostic-only step tag used to keep matched prefill/decode tensor dumps distinct. A negative
// step disables the suffix and preserves the existing D1.1/D1.2 filenames.
void set_matched_diagnostic_step(std::int32_t step) noexcept;
[[nodiscard]] std::string matched_diagnostic_step_suffix();

void bf16_to_fp32(const Tensor& input, Tensor& output, cudaStream_t stream);
void fp32_to_bf16(const Tensor& input, Tensor& output, cudaStream_t stream);

} // namespace ninfer::targets::qwen3_6::oscar_internal
