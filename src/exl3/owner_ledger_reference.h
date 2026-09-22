#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::exl3 {

// Independent, bounded ownership oracle for prepared tests. It deliberately
// does not consume Exl3ResourceInventory records or candidate traversal output.
class Exl3IndependentOwnerLedgerReference {
public:
    enum class Domain : std::uint8_t { host, cuda_registered_host, device, count };
    enum class ReferenceKind : std::uint8_t { shared_alias, view };
    enum class Cleanup : std::uint8_t { success, failure, callback_lost };

    static constexpr std::size_t maximum_allocations=16;
    static constexpr std::size_t maximum_references=32;

    struct Snapshot {
        std::array<std::size_t,static_cast<std::size_t>(Domain::count)> bytes{};
        std::size_t allocations=0;
        std::size_t aliases=0;
        std::size_t views=0;
        std::size_t quarantined=0;

        [[nodiscard]] std::size_t total_bytes() const {
            std::size_t total=0;
            for(const auto value:bytes)total=checked_add(total,value);
            return total;
        }
    private:
        static std::size_t checked_add(std::size_t left,std::size_t right) {
            if(right>std::numeric_limits<std::size_t>::max()-left)
                throw std::overflow_error("owner-ledger reference total overflows size_t");
            return left+right;
        }
    };

    void allocate(std::uint64_t allocation,std::uint64_t owner,Domain domain,
                  std::size_t bytes) {
        if(!allocation || !owner || !bytes || domain==Domain::count)
            throw std::invalid_argument("owner-ledger reference allocation is invalid");
        if(find_allocation(allocation))
            throw std::logic_error("owner-ledger reference allocation identity reused");
        auto* slot=free_allocation();
        if(!slot)throw std::length_error("owner-ledger reference allocation capacity exceeded");
        *slot={allocation,owner,domain,bytes,true,false,false};
    }

    void reference(std::uint64_t reference_id,std::uint64_t allocation,
                   ReferenceKind kind) {
        if(!reference_id || reference_id_known(reference_id))
            throw std::logic_error("owner-ledger reference identity reused");
        auto* target=find_allocation(allocation);
        if(!target || !target->charged || target->retiring || target->quarantined)
            throw std::logic_error("owner-ledger late reference has no live owner");
        auto* slot=free_reference();
        if(!slot)throw std::length_error("owner-ledger reference capacity exceeded");
        *slot={reference_id,allocation,kind,true};
    }

    void retire(std::uint64_t allocation,Cleanup cleanup) {
        auto* target=require_allocation(allocation);
        if(target->retiring || target->quarantined)
            throw std::logic_error("owner-ledger allocation retired twice");
        if(cleanup!=Cleanup::success) {
            target->quarantined=true;
            return;
        }
        target->retiring=true;
        release_if_unreferenced(*target);
    }

    void release_reference(std::uint64_t reference_id) {
        auto* handle=find_reference(reference_id);
        if(!handle)throw std::logic_error("owner-ledger reference release is stale");
        const auto allocation=handle->allocation;
        handle->active=false;
        auto* target=require_allocation(allocation);
        if(target->retiring && !target->quarantined)release_if_unreferenced(*target);
    }

    [[nodiscard]] Snapshot snapshot() const {
        Snapshot result;
        for(const auto& allocation:allocations_)if(allocation.charged) {
            const auto domain=static_cast<std::size_t>(allocation.domain);
            if(allocation.bytes>std::numeric_limits<std::size_t>::max()-result.bytes[domain])
                throw std::overflow_error("owner-ledger reference domain overflows size_t");
            result.bytes[domain]+=allocation.bytes;
            ++result.allocations;
            if(allocation.quarantined)++result.quarantined;
        }
        for(const auto& handle:references_)if(handle.active) {
            if(handle.kind==ReferenceKind::shared_alias)++result.aliases;
            else ++result.views;
        }
        return result;
    }

    [[nodiscard]] bool charged(std::uint64_t allocation) const noexcept {
        const auto* value=find_allocation(allocation);
        return value && value->charged;
    }

private:
    struct Allocation {
        std::uint64_t id=0;
        std::uint64_t owner=0;
        Domain domain=Domain::host;
        std::size_t bytes=0;
        bool charged=false;
        bool retiring=false;
        bool quarantined=false;
    };
    struct Reference {
        std::uint64_t id=0;
        std::uint64_t allocation=0;
        ReferenceKind kind=ReferenceKind::shared_alias;
        bool active=false;
    };

    Allocation* find_allocation(std::uint64_t id) noexcept {
        for(auto& value:allocations_)if(value.id==id)return &value;
        return nullptr;
    }
    const Allocation* find_allocation(std::uint64_t id) const noexcept {
        for(const auto& value:allocations_)if(value.id==id)return &value;
        return nullptr;
    }
    Allocation* require_allocation(std::uint64_t id) {
        auto* value=find_allocation(id);
        if(!value || !value->charged)
            throw std::logic_error("owner-ledger allocation is not charged");
        return value;
    }
    Reference* find_reference(std::uint64_t id) noexcept {
        for(auto& value:references_)if(value.active && value.id==id)return &value;
        return nullptr;
    }
    [[nodiscard]] bool reference_id_known(std::uint64_t id) const noexcept {
        for(const auto& value:references_)if(value.id==id)return true;
        return false;
    }
    Allocation* free_allocation() noexcept {
        for(auto& value:allocations_)if(!value.id)return &value;
        return nullptr;
    }
    Reference* free_reference() noexcept {
        for(auto& value:references_)if(!value.id)return &value;
        return nullptr;
    }
    [[nodiscard]] bool has_reference(std::uint64_t allocation) const noexcept {
        for(const auto& value:references_)
            if(value.active && value.allocation==allocation)return true;
        return false;
    }
    void release_if_unreferenced(Allocation& allocation) noexcept {
        if(!has_reference(allocation.id))allocation.charged=false;
    }

    std::array<Allocation,maximum_allocations> allocations_{};
    std::array<Reference,maximum_references> references_{};
};

} // namespace ninfer::exl3
