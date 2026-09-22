#pragma once
#include "exl3/device_page_storage.h"
#include "exl3/registered_kv_upload.h"

namespace ninfer::exl3 {
// Exact ordinary host-page fill. The initial command waits for readiness before
// returning; lookup peers can use host fallback while this producer is active.
inline void exl3_fill_device_page(const Exl3DevicePageStorage::Prepared& prepared,
    cudaStream_t stream,cudaEvent_t event,std::shared_ptr<const void> event_owner,
    std::uint64_t acquisition,std::uint64_t execution,Exl3KVUploadProvider provider={},
    std::optional<Exl3KVRegistrationCache::ExternalRead> registration_read={}) {
    if(!prepared.fill || !prepared.fill.use_count() || !prepared.fill->owns_destination(prepared.destination) ||
        !prepared.fill->key().current() || !event || !acquisition || !execution ||
        !event_owner || !event_owner.use_count() || !provider.copy || !provider.record || !provider.wait)
        throw std::invalid_argument("device page fill command storage/provider");
    const auto& page=prepared.fill->key().page();
    // Preflight every plane before submitting any transfer. Partial preparation
    // cannot enqueue a prefix and then discover a missing later source plane.
    std::array<std::optional<Exl3ExactKVExtent>,32> source;
    for(int bank=0;bank<16;++bank)for(int plane=0;plane<2;++plane)
        source[bank*2+plane].emplace(Exl3ExactKVExtent::view(page,bank,
            plane==0?Exl3ExactKVExtent::Plane::key:Exl3ExactKVExtent::Plane::value,page->first+64));
    const auto generation=prepared.fill->begin(acquisition,execution,
        reinterpret_cast<std::uintptr_t>(event),std::move(event_owner),std::move(registration_read));
    const auto accept=[&](cudaError_t error) {
        if(error==cudaSuccess)return;
        prepared.fill->finish(generation,static_cast<int>(error));
        throw std::runtime_error("device page fill transfer/readiness failed");
    };
    try {
        for(int bank=0;bank<16;++bank)for(int plane=0;plane<2;++plane) {
            const auto index=bank*2+plane;
            accept(provider.copy(prepared.destination.data()+index*Exl3DevicePageFill::plane_elements,
                source[index]->data(),source[index]->bytes(),cudaMemcpyHostToDevice,stream));
            prepared.fill->plane_submitted(generation,bank,plane==0);
        }
        accept(provider.record(event,stream));
        accept(provider.wait(event));
        if(!prepared.fill->finish(generation,0) || !prepared.fill->ready())
            throw std::logic_error("device page fill final readiness scope changed");
    } catch(...) {prepared.fill->abandon_producer();throw;}
}
}
