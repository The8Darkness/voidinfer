#pragma once

#include <ninfer/targets/qwen3_6/vision_control.h>
#include "exl3/retained_descriptor_ledger.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace ninfer::exl3 {

inline bool exl3_same_vision_control(
    const targets::qwen3_6::VisionItemControl& left,
    const targets::qwen3_6::VisionItemControl& right) noexcept {
    return left.modality==right.modality &&
        left.grid.temporal==right.grid.temporal &&
        left.grid.height==right.grid.height && left.grid.width==right.grid.width &&
        left.patch_begin==right.patch_begin && left.patch_count==right.patch_count &&
        left.merged_count==right.merged_count &&
        left.segment_length==right.segment_length && left.segment_count==right.segment_count &&
        left.position_ids==right.position_ids && left.scatter_indices==right.scatter_indices &&
        left.position_table_indices==right.position_table_indices &&
        left.position_table_weights==right.position_table_weights;
}

struct Exl3EncodedMediaResult {
    std::vector<float> embeddings;
    std::size_t patch_count = 0;
    std::size_t merged_count = 0;
    std::optional<RetainedHostAllocationLedger::Ticket> retention_credit;
    std::optional<RetainedHostAllocationLedger::Ticket> external_retention_credit;
};

struct Exl3EncodedMediaRetentionCredits {
    std::optional<RetainedHostAllocationLedger::Ticket> replay;
    std::optional<RetainedHostAllocationLedger::Ticket> output;
};
using Exl3EncodedMediaRetentionReserve=std::function<Exl3EncodedMediaRetentionCredits(
    std::uint64_t replay_bytes,std::uint64_t output_bytes)>;

class Exl3EncodedMediaEntry {
public:
    enum class Completion : std::uint8_t { pending,ready,failed };

    Exl3EncodedMediaEntry(
        std::shared_ptr<const targets::qwen3_6::PreparedMediaPayload> payload,
        targets::qwen3_6::VisionItem item,
        std::shared_ptr<const void> encoder,
        targets::qwen3_6::VisionItemControl control,
        std::optional<RetainedHostAllocationLedger::Ticket> replay_credit={},
        std::optional<RetainedHostAllocationLedger::Ticket> output_credit={},
        std::optional<RetainedHostAllocationLedger::Ticket> external_replay_credit={},
        std::optional<RetainedHostAllocationLedger::Ticket> external_output_credit={})
        : payload_(std::move(payload)),item_(std::move(item)),encoder_(std::move(encoder)),
          control_(std::move(control)),replay_credit_(std::move(replay_credit)),
          output_credit_(std::move(output_credit)),
          external_replay_credit_(std::move(external_replay_credit)),
          external_output_credit_(std::move(external_output_credit)) {
        if(!identity_valid())
            throw std::invalid_argument("encoded media entry identity/geometry");
        if((replay_credit_ && replay_credit_->bytes()!=replay_bytes()) ||
           (output_credit_ && output_credit_->bytes()!=output_bytes()) ||
           (external_replay_credit_ && external_replay_credit_->bytes()!=replay_bytes()) ||
           (external_output_credit_ && external_output_credit_->bytes()!=output_bytes()))
            throw std::invalid_argument("encoded media entry retention credit extent");
    }

    [[nodiscard]] bool matches(
        const std::shared_ptr<const targets::qwen3_6::PreparedMediaPayload>& payload,
        const targets::qwen3_6::VisionItem& item,
        const std::shared_ptr<const void>& encoder,
        const targets::qwen3_6::VisionItemControl& control) const noexcept {
        return payload && encoder && payload_.get()==payload.get() &&
            encoder_.get()==encoder.get() && exl3_same_vision_control(control_,control) &&
            payload->preprocess==payload_->preprocess && payload->storage==payload_->storage &&
            item.modality==item_.modality && item.grid.temporal==item_.grid.temporal &&
            item.grid.height==item_.grid.height && item.grid.width==item_.grid.width &&
            item.patch_begin==item_.patch_begin && item.patch_count==item_.patch_count &&
            item.content_digest==item_.content_digest && item.preprocess==item_.preprocess &&
            item.timestamps==item_.timestamps && item.token_spans.size()==item_.token_spans.size() &&
            std::equal(item.token_spans.begin(),item.token_spans.end(),item_.token_spans.begin(),
                [](const auto& left,const auto& right) {
                    return left.begin==right.begin && left.count==right.count;
                }) && item.patch_storage==item_.patch_storage;
    }

    void publish(std::vector<float> embeddings) {
        const auto expected=control_.merged_count*5120ULL;
        if(embeddings.size()!=expected ||
           std::any_of(embeddings.begin(),embeddings.end(),
               [](float value){return !std::isfinite(value);}))
            throw std::invalid_argument("encoded media result extent/finiteness");
        auto ready=std::make_shared<const Exl3EncodedMediaResult>(
            Exl3EncodedMediaResult{std::move(embeddings),control_.patch_count,
                control_.merged_count,std::move(output_credit_),
                std::move(external_output_credit_)});
        {
            std::lock_guard lock(mutex_);
            if(completion_!=Completion::pending)
                throw std::logic_error("encoded media entry completion already published");
            result_=std::move(ready);completion_=Completion::ready;
        }
        changed_.notify_all();
    }

    void fail() noexcept {
        {
            std::lock_guard lock(mutex_);
            if(completion_!=Completion::pending)return;
            completion_=Completion::failed;
        }
        changed_.notify_all();
    }

    [[nodiscard]] Completion completion() const noexcept {
        std::lock_guard lock(mutex_);return completion_;
    }

    [[nodiscard]] std::shared_ptr<const Exl3EncodedMediaResult> ready_result() const {
        std::lock_guard lock(mutex_);
        if(completion_!=Completion::ready || !result_)
            throw std::logic_error("encoded media entry is not ready");
        return result_;
    }

    [[nodiscard]] std::shared_ptr<const Exl3EncodedMediaResult> wait_ready(
        const std::function<bool()>& cancelled={}) const {
        std::unique_lock lock(mutex_);
        while(completion_==Completion::pending) {
            if(cancelled && cancelled())
                throw std::runtime_error("encoded media cache wait cancelled");
            changed_.wait_for(lock,std::chrono::milliseconds(10));
        }
        if(cancelled && cancelled())
            throw std::runtime_error("encoded media cache wait cancelled");
        if(completion_!=Completion::ready || !result_)
            throw std::runtime_error("encoded media producer failed");
        return result_;
    }

    [[nodiscard]] const std::shared_ptr<const targets::qwen3_6::PreparedMediaPayload>&
    payload_owner() const noexcept {return payload_;}
    [[nodiscard]] const targets::qwen3_6::VisionItem& item() const noexcept {return item_;}
    [[nodiscard]] const std::shared_ptr<const void>& encoder_owner() const noexcept {
        return encoder_;
    }
    [[nodiscard]] const targets::qwen3_6::VisionItemControl& control() const noexcept {
        return control_;
    }
    [[nodiscard]] std::size_t replay_bytes() const noexcept {
        return sizeof(targets::qwen3_6::PreparedMediaPayload)+
            payload_->patch_elements*sizeof(std::uint16_t);
    }
    [[nodiscard]] std::size_t output_bytes() const noexcept {
        return control_.merged_count*5120ULL*sizeof(float);
    }
    [[nodiscard]] bool replay_credit_belongs_to(
        const RetainedHostAllocationLedger& ledger) const noexcept {
        return replay_credit_ && ledger.owns(*replay_credit_) &&
            replay_credit_->bytes()==replay_bytes();
    }

private:
    [[nodiscard]] bool identity_valid() const noexcept {
        if(!payload_ || !encoder_ || !payload_->patches ||
           payload_->storage!=VisionPatchStorage::Float16 ||
           payload_->patch_elements>std::numeric_limits<std::size_t>::max()/sizeof(std::uint16_t) ||
           control_.merged_count>std::numeric_limits<std::size_t>::max()/(5120ULL*sizeof(float)) ||
           item_.patch_storage!=payload_->storage || item_.preprocess!=payload_->preprocess ||
           item_.modality!=control_.modality ||
           item_.grid.temporal!=control_.grid.temporal ||
           item_.grid.height!=control_.grid.height || item_.grid.width!=control_.grid.width ||
           item_.patch_begin!=control_.patch_begin || item_.patch_count!=control_.patch_count ||
           !payload_->preprocess.valid(
               control_.modality==targets::qwen3_6::PromptModality::Video) ||
           !control_.patch_count || control_.patch_count%4 ||
           payload_->patch_elements!=
               control_.patch_count*targets::qwen3_6::kPreparedVisionPatchFeatures ||
           control_.merged_count!=control_.patch_count/4 ||
           control_.grid.temporal<=0 || control_.grid.height<=0 ||
           control_.grid.width<=0 || control_.grid.height%2 || control_.grid.width%2 ||
           static_cast<std::uint64_t>(control_.grid.temporal)*
               static_cast<std::uint64_t>(control_.grid.height)*
           static_cast<std::uint64_t>(control_.grid.width)!=control_.patch_count)
            return false;
        const bool video=control_.modality==targets::qwen3_6::PromptModality::Video;
        if(video) {
            if(item_.timestamps.size()!=static_cast<std::size_t>(control_.segment_count) ||
               item_.token_spans.size()!=static_cast<std::size_t>(control_.segment_count))return false;
            for(std::size_t i=0;i<item_.timestamps.size();++i)
                if(!std::isfinite(item_.timestamps[i]) || item_.timestamps[i]<0.0 ||
                   (i && item_.timestamps[i]<=item_.timestamps[i-1]))return false;
        } else if(!item_.timestamps.empty() || item_.token_spans.size()!=1)return false;
        const auto per_segment=control_.merged_count/static_cast<std::size_t>(control_.segment_count);
        std::size_t scatter=0;
        for(std::size_t i=0;i<item_.token_spans.size();++i) {
            if(item_.token_spans[i].count!=per_segment ||
               (i && item_.token_spans[i].begin<item_.token_spans[i-1].begin+
                    item_.token_spans[i-1].count))return false;
            for(std::size_t offset=0;offset<item_.token_spans[i].count;++offset,++scatter)
                if(scatter>=control_.scatter_indices.size() ||
                   control_.scatter_indices[scatter]!=static_cast<std::int32_t>(
                       item_.token_spans[i].begin+offset))return false;
        }
        if(scatter!=control_.scatter_indices.size())return false;
        return control_.segment_length==control_.grid.height*control_.grid.width &&
            control_.segment_count==control_.grid.temporal &&
            control_.position_ids.size()==control_.patch_count*2 &&
            control_.position_table_indices.size()==control_.patch_count*4 &&
            control_.position_table_weights.size()==control_.patch_count*4;
    }

    std::shared_ptr<const targets::qwen3_6::PreparedMediaPayload> payload_;
    targets::qwen3_6::VisionItem item_;
    std::shared_ptr<const void> encoder_;
    targets::qwen3_6::VisionItemControl control_;
    mutable std::mutex mutex_;
    mutable std::condition_variable changed_;
    Completion completion_=Completion::pending;
    std::shared_ptr<const Exl3EncodedMediaResult> result_;
    std::optional<RetainedHostAllocationLedger::Ticket> replay_credit_;
    std::optional<RetainedHostAllocationLedger::Ticket> output_credit_;
    std::optional<RetainedHostAllocationLedger::Ticket> external_replay_credit_;
    std::optional<RetainedHostAllocationLedger::Ticket> external_output_credit_;
};

} // namespace ninfer::exl3
