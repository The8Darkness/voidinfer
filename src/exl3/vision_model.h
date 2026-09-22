#pragma once
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "exl3/encoded_media_cache.h"
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <vector>
#include <cstdint>

namespace ninfer::exl3 {
struct Exl3VisionBoundaryCode {
    static constexpr int block_operator(int block,int boundary) noexcept {
        return 1000+block*16+boundary;
    }
    static constexpr int block_residual(int block) noexcept {return block;}
    static constexpr int merger_normalized=2000;
    static constexpr int merger_up=2001;
    static constexpr int merger_output=2002;
};
struct Exl3VisionStats {
    std::size_t model_device_bytes=0, context_device_bytes=0;
    std::size_t patches=0, merged_tokens=0, packed_projections=0;
    double wall_ms=0;
    std::size_t temporary_host_bytes=0;
};
struct Exl3VisionOutput {
    // FP32 merger output; immutable host ownership after completion fence.
    std::vector<float> embeddings;
    Exl3VisionStats stats;
};
struct Exl3EncodedMediaResult;
struct Exl3EncodedMediaCacheLimits {
    std::size_t max_entries=4;
    std::uint64_t max_encoded_host_bytes=64ULL<<20;
    std::uint64_t max_replay_host_bytes=64ULL<<20;
};
struct Exl3EncodedMediaCacheStats {
    std::uint64_t hits=0,misses=0,cancelled_consumers=0,failed_producers=0;
    std::uint64_t evictions=0,quota_refusals=0;
    std::size_t entries=0;
    std::uint64_t retained_encoded_host_bytes=0,retained_replay_host_bytes=0;
    std::uint64_t encoder_scratch_device_bytes=0,inflight_scratch_device_bytes=0;
    std::size_t inflight_producers=0;
};
class Exl3VisionContext;
// Explicit research API only. No Engine policy selects it until operator and
// complete media/L2 qualification. Packed weights remain the source of truth.
class Exl3VisionModel {
public:
    explicit Exl3VisionModel(const std::filesystem::path& directory);
    ~Exl3VisionModel();
    Exl3VisionModel(const Exl3VisionModel&)=delete;
    Exl3VisionModel& operator=(const Exl3VisionModel&)=delete;
    std::size_t device_bytes() const noexcept;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class Exl3VisionContext;
};
class Exl3VisionContext {
public:
    explicit Exl3VisionContext(const Exl3VisionModel& model, int maximum_patches=1024,
        Exl3EncodedMediaCacheLimits cache_limits={});
    ~Exl3VisionContext();
    Exl3VisionContext(const Exl3VisionContext&)=delete;
    Exl3VisionContext& operator=(const Exl3VisionContext&)=delete;
    // Video temporal segments execute independently using the same image extent;
    // an observer is called separately per frame. One context/stream per caller;
    // model weights may be shared. No output is
    // returned on cancellation. A failed stream fence permanently poisons reuse.
    Exl3VisionOutput encode_numeric_candidate(
        const targets::qwen3_6::PreparedMediaPayload& payload,
        const targets::qwen3_6::VisionItemControl& control,
        const std::function<bool()>& cancelled={},
        const std::function<void(int,std::span<const float>)>& layer_observer={});
    std::shared_ptr<const Exl3EncodedMediaResult> encode_prepared_cached(
        std::shared_ptr<const targets::qwen3_6::PreparedMediaPayload> payload,
        const targets::qwen3_6::VisionItem& item,
        const targets::qwen3_6::VisionItemControl& control,
        const std::function<bool()>& cancelled={},
        const Exl3EncodedMediaRetentionReserve& reserve={});
    [[nodiscard]] Exl3EncodedMediaCacheStats encoded_media_cache_stats() const;
    void set_encoded_media_cache_limits(Exl3EncodedMediaCacheLimits limits);
    std::size_t device_bytes() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ninfer::exl3
