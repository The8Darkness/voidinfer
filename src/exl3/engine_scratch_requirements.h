#pragma once
#include "exl3/resource_inventory.h"

namespace ninfer::exl3 {
// The pinned Engine tap allocation menu. Used by both admission and allocation;
// it describes these five private destinations, not context/draft workspaces.
struct Exl3EngineTapRequirements {
    static constexpr unsigned layers=5,rows=16,columns=5120;
    static constexpr std::uint64_t plane_bytes=std::uint64_t(rows)*columns*sizeof(std::uint16_t);
    static Exl3ResourceInventory::Requirement for_lanes(unsigned lanes,std::uint64_t owner_metadata_bytes=0) {
        if(lanes!=1 && lanes!=2)
            throw std::invalid_argument("Engine tap requirement supports physical C1/C2");
        Exl3ResourceInventory::Requirement result;
        result.configuration=lanes;
        result.add(Exl3ResourceInventory::Domain::device,lanes,layers*plane_bytes);
        if(owner_metadata_bytes)result.add(Exl3ResourceInventory::Domain::host_metadata,lanes,owner_metadata_bytes);
        return result;
    }
};
}
