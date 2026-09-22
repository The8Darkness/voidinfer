#pragma once
#include "exl3/vericache_request.h"

namespace ninfer::exl3 {
// A supplied projection root may be the acquired root or one verified private
// transition from it. This is a conditioning check, not a lease-authority check.
inline bool shared_conditioning_scope_valid(
    const std::shared_ptr<const Exl3VeriCacheRequest>& root,
    const std::shared_ptr<const Exl3VeriCacheRequest>& acquired,
    bool draft,int position) noexcept {
    if(!draft || !root || !root.use_count() || !acquired || !acquired.use_count() ||
       !root->state() || root->state()->position()!=position)return false;
    const bool same=root.get()==acquired.get() &&
        !root.owner_before(acquired) && !acquired.owner_before(root);
    return same || root->is_child_of(*acquired);
}
}
