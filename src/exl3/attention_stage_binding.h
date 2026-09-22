#pragma once
#include "exl3/attention_stage.h"
#include "exl3/attention_stage_storage.h"

namespace ninfer::exl3 {
struct Exl3AttentionStageBinding {
    Exl3AttentionStage::Ticket ticket;
    std::uint16_t* destination=nullptr;
    cudaEvent_t producer=nullptr,consumer=nullptr;
};
// All physical addresses derive from the same retained, reserved owner. Extent
// bounds and representation are checked before changing the stage generation.
inline Exl3AttentionStageBinding exl3_bind_attention_stage(Exl3AttentionStage& stages,
    Exl3ExactKVExtent extent,const std::shared_ptr<Exl3AttentionStageStorage>& storage,
    std::uint64_t acquisition,std::uint64_t execution,std::shared_ptr<const void> source_group={},
    std::optional<Exl3KVRegistrationCache::ExternalRead> registration_read={}) {
    if(!storage || !storage.use_count() || !extent.rows() || !extent.backing_current())
        throw std::invalid_argument("attention stage missing storage/current extent");
    const auto offset=extent.destination_offset(storage->capacity());
    const unsigned plane=extent.plane()==Exl3ExactKVExtent::Plane::key?0U:1U;
    auto* destination=storage->plane(plane)+offset;
    const auto producer=storage->producer_event(plane),consumer=storage->consumer_event(plane);
    auto ticket=stages.begin(plane,std::move(extent),storage,acquisition,execution,
        reinterpret_cast<std::uintptr_t>(producer),reinterpret_cast<std::uintptr_t>(consumer),std::move(source_group),
        std::move(registration_read));
    return {ticket,destination,producer,consumer};
}
}
