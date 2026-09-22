#pragma once
#include "exl3/resource_inventory.h"
#include <cstddef>
#include <iterator>
#include <type_traits>

namespace ninfer::exl3 {
inline bool exl3_host_kv_append_download_interval(int first,int rows,int position,int capacity,
    int& planned_end) noexcept {
    if(capacity<=0 || position<0 || position>capacity || first<0 || first>capacity ||
        rows<=0 || rows>64 || rows>capacity-first)return false;
    const int end=first+rows;
    if(end<=position)return true; // No rows from this page are downloaded.
    const int begin=first<position?position:first;
    if(planned_end>=0 && begin!=planned_end)return false;
    planned_end=end;return true;
}
// Reject either truncated plane without advancing the accepted page sequence.
inline bool exl3_host_kv_append_download_page(int first,int rows,int position,int capacity,
    std::size_t key_elements,std::size_t value_elements,int& planned_end) noexcept {
    int next=planned_end;
    if(!exl3_host_kv_append_download_interval(first,rows,position,capacity,next))return false;
    const auto required=static_cast<std::size_t>(rows)*1024;
    if(key_elements<required || value_elements<required)return false;
    planned_end=next;
    return true;
}
// The range and its owners must stay unchanged across both passes. Apply must
// not fail: this provides preflight atomicity, not rollback of destination writes.
template<class Range,class Validate,class Apply>
void exl3_host_kv_scatter_batch(const Range& pieces,Validate&& validate,Apply&& apply) {
    using Piece=decltype(*std::begin(pieces));
    static_assert(std::is_nothrow_invocable_v<Apply&,Piece>,
        "scatter application must not throw after destination writes begin");
    for(const auto& piece:pieces)validate(piece);
    for(const auto& piece:pieces)apply(piece);
}
inline bool exl3_host_kv_scatter_fits(std::size_t destination_bytes,std::size_t destination_offset,
    std::size_t staging_bytes,std::size_t staging_offset,std::size_t bytes) noexcept {
    return bytes && destination_offset<=destination_bytes && bytes<=destination_bytes-destination_offset &&
        staging_offset<=staging_bytes && bytes<=staging_bytes-staging_offset;
}
struct Exl3HostKVTransferRequirements {
    std::size_t descriptors=0;
    // Object metadata is charged by the caller using its concrete owner type.
    std::size_t pinned_owners=0;
    Exl3ResourceInventory::Requirement resources;
    static Exl3HostKVTransferRequirements derive(int capacity,bool batch_copy,
        bool pinned_chunks,bool pinned_d2h,std::size_t chunk_bytes,
        bool banked_d2h=false,std::size_t banked_bytes=0,
        std::size_t pinned_slots=2) {
        if(capacity<=0)throw std::invalid_argument("host KV transfer context extent");
        if(pinned_d2h && !pinned_chunks)throw std::invalid_argument("host KV pinned D2H requires chunks");
        Exl3HostKVTransferRequirements plan;
        if(!batch_copy)return plan;
        if(banked_d2h && !pinned_d2h)
            throw std::invalid_argument("host KV banked D2H requires pinned D2H");
        using Domain=Exl3ResourceInventory::Domain;
        if(pinned_chunks) {
            if(!chunk_bytes || pinned_slots<2)
                throw std::invalid_argument("host KV pinned chunk extent/slots");
            plan.pinned_owners=pinned_slots;
            plan.resources.add(Domain::cuda_registered_host,plan.pinned_owners,chunk_bytes);
        }
        if(banked_d2h) {
            if(!banked_bytes)throw std::invalid_argument("host KV banked D2H extent");
            ++plan.pinned_owners;
            plan.resources.add(Domain::cuda_registered_host,1,banked_bytes);
        }
        if(!pinned_chunks || !pinned_d2h) {
            plan.descriptors=(static_cast<std::size_t>(capacity)/64+(capacity%64!=0))*2;
            plan.resources.add(Domain::host_metadata,plan.descriptors,
                sizeof(void*)+sizeof(const void*)+sizeof(std::size_t));
        }
        return plan;
    }
};
}
