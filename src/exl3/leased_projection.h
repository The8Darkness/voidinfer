#pragma once
#include "exl3/packed_projection.h"
#include "exl3/vericache_serving_coordinator.h"

namespace ninfer::exl3 {
// Bridge from existing admission authority into a compute descriptor. Keep the
// lease array and coordinator alive until the dispatch completes. Physical
// execution generation/buffers remain the exclusive execution owner's facts.
inline Exl3PackedProjectionPlan assemble_exl3_leased_projection(
    const Exl3VeriCacheServingCoordinator& coordinator,
    std::span<const Exl3VeriCacheServingCoordinator::Lease> leases,
    std::span<const Exl3ProjectionRows> physical_rows,int capacity) {
    if(physical_rows.size()<2 || physical_rows.size()>8 || capacity<2 || capacity>16 ||
       leases.size()!=physical_rows.size() || !coordinator.compute_leases_current(leases))
        throw std::invalid_argument("shared projection acquisition snapshot");
    std::array<Exl3ProjectionRows,8> rows;
    for(std::size_t i=0;i<physical_rows.size();++i) {
        rows[i]=physical_rows[i];
        const auto& lease=leases[i];auto& row=rows[i];
        if(row.position!=lease.root->state()->position())
            throw std::invalid_argument("shared projection root position");
        // Unbound descriptors may acquire authority here. Already-bound rows
        // must never silently become another request's physical work.
        if((row.request && row.request!=lease.ticket.request_id) ||
           (row.acquisition && row.acquisition!=lease.acquisition) ||
           (row.root && !Exl3ProjectionRows::same_owner(row.root,lease.root)) ||
           (row.model && !Exl3ProjectionRows::same_owner(row.model,lease.root->state()->model_identity())))
            throw std::invalid_argument("shared projection physical authority mismatch");
        row.request=lease.ticket.request_id;row.acquisition=lease.acquisition;
        row.root=lease.root;row.model=lease.root->state()->model_identity();
    }
    return Exl3PackedProjectionPlan::assemble(std::span(rows).first(physical_rows.size()),capacity,[&](const auto&){
        return coordinator.compute_leases_current(leases);
    });
}
} // namespace ninfer::exl3
