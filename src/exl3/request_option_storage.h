#pragma once
#include "ninfer/types.h"
#include "exl3/resource_inventory.h"

namespace ninfer::exl3 {
// Fixed option objects belong to Request::Storage. Count dynamic capacities,
// conservatively including inline string bytes; exclude allocator overhead.
inline std::uint64_t request_stop_storage_bytes(const StopPolicy& stop) {
    Exl3ResourceInventory::Requirement required;
    constexpr auto host=Exl3ResourceInventory::Domain::host_metadata;
    if(stop.token_ids.capacity())required.add(host,stop.token_ids.capacity(),sizeof(TokenId));
    if(stop.strings.capacity())required.add(host,stop.strings.capacity(),sizeof(StopString));
    for(const auto& value:stop.strings) {
        if(value.text.capacity())required.add(host,value.text.capacity(),1);
        required.add(host,1,1);
    }
    return required.units[static_cast<unsigned>(host)];
}
}
