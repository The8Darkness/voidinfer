#pragma once
#include "exl3/resource_inventory.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace ninfer::exl3 {
// Driver allocation extents must be supplied by a supported backend contract.
// Zero is a declared extent only when known=true; missing observations never
// become a zero-cost capture. Existing graph owners remain charged during capture.
struct Exl3GraphCaptureExtent {
    Exl3ResourceInventory::Totals retained{};
    Exl3ResourceInventory::Totals temporary{};
    bool known=false;
};
enum class Exl3GraphCaptureDisposition : std::uint8_t {
    capture,
    eager_fallback
};
struct Exl3GraphCaptureEntry {
    unsigned rows=0;
    Exl3GraphCaptureExtent extent;
    Exl3GraphCaptureDisposition disposition=Exl3GraphCaptureDisposition::capture;
};
struct Exl3GraphCaptureRequirements {
    Exl3ResourceInventory::Requirement additional_peak;
    Exl3ResourceInventory::Totals retained_after_capture{};
    Exl3ResourceInventory::Totals peak_with_old_owners{};
    static Exl3GraphCaptureRequirements derive(std::uint64_t configuration,
        const Exl3ResourceInventory::Totals& old_owners,
        std::span<const Exl3GraphCaptureEntry> entries) {
        using Inventory=Exl3ResourceInventory;
        if(!configuration || entries.empty() || entries.size()>4)
            throw std::invalid_argument("graph capture menu identity or capacity");
        Exl3GraphCaptureRequirements result;
        result.additional_peak.configuration=configuration;
        Inventory::Requirement retained,temporary,peak;
        peak.units=old_owners;
        unsigned seen=0,captured=0;
        for(const auto& entry:entries) {
            const unsigned bit=entry.rows==1?1:entry.rows==4?2:entry.rows==6?4:entry.rows==8?8:0;
            if(!bit || (seen&bit) || !entry.extent.known)
                throw std::invalid_argument("graph capture unknown extent or unsupported menu");
            seen|=bit;
            const auto graphs=static_cast<unsigned>(Inventory::Domain::graph_count);
            if(entry.disposition==Exl3GraphCaptureDisposition::eager_fallback) {
                if(entry.extent.retained!=Inventory::Totals{} ||
                   entry.extent.temporary!=Inventory::Totals{})
                    throw std::invalid_argument(
                        "graph fallback entry declares captured resources");
                continue;
            }
            if(entry.extent.retained[graphs]<2)
                throw std::invalid_argument("graph capture omits definition/executable ownership");
            ++captured;
            for(unsigned i=0;i<static_cast<unsigned>(Inventory::Domain::count);++i) {
                const auto domain=static_cast<Inventory::Domain>(i);
                if(entry.extent.retained[i])retained.add(domain,1,entry.extent.retained[i]);
                // Capture entries are serialized. Previous entries persist, but
                // their temporary construction storage is not concurrently live.
                temporary.units[i]=std::max(temporary.units[i],entry.extent.temporary[i]);
            }
        }
        if(!captured)throw std::invalid_argument("graph capture menu has no captured entry");
        result.retained_after_capture=retained.units;
        result.additional_peak.units=retained.units;
        for(unsigned i=0;i<static_cast<unsigned>(Inventory::Domain::count);++i) {
            const auto domain=static_cast<Inventory::Domain>(i);
            if(temporary.units[i])result.additional_peak.add(domain,1,temporary.units[i]);
            if(result.additional_peak.units[i])peak.add(domain,1,result.additional_peak.units[i]);
        }
        result.peak_with_old_owners=peak.units;
        return result;
    }
};

// Reserve the complete serialized capture peak as a retained menu token. The
// temporary maximum intentionally remains charged: a later capture or recapture
// must not depend on memory/resources becoming available after startup.
template<class Coordinator>
Exl3ResourceInventory::Totals exl3_reserve_graph_menu_startup(
    Coordinator& authority,std::uint64_t configuration,
    std::span<const Exl3GraphCaptureEntry> entries,
    std::shared_ptr<const void> owner,std::uint64_t first_slot) {
    if(!owner)throw std::invalid_argument("graph menu reservation owner");
    const auto plan=Exl3GraphCaptureRequirements::derive(configuration,{},entries);
    const auto reservation=plan.additional_peak;
    authority.allocate_startup_resources(reservation,[&](std::uint64_t accepted) {
        if(accepted!=configuration)
            throw std::logic_error("graph menu reservation identity");
        Exl3ResourceInventory actual;
        for(unsigned i=0;i<reservation.units.size();++i)
            if(reservation.units[i])actual.add({owner,first_slot+i,
                static_cast<Exl3ResourceInventory::Domain>(i),reservation.units[i]});
        return actual;
    });
    return reservation.units;
}
// Each factory invocation must return idle retained owners after releasing its
// temporary construction storage. No asynchronous upload/launch may escape it.
// Old owners already belong to the coordinator; this transaction adds only new
// owners and never credits an old graph merely because replacement was selected.
template<class Coordinator,class Factory,class Rollback,class Observe=std::nullptr_t>
Exl3ResourceInventory exl3_allocate_graph_menu_startup(Coordinator& authority,
    std::uint64_t configuration,std::span<const Exl3GraphCaptureEntry> entries,
    Factory&& factory,Rollback&& rollback,Observe&& observe=nullptr) {
    static_assert(std::is_nothrow_invocable_v<Rollback&>,"graph menu rollback must not throw");
    const auto plan=Exl3GraphCaptureRequirements::derive(configuration,{},entries);
    std::array<Exl3GraphCaptureEntry,4> menu{};
    std::copy(entries.begin(),entries.end(),menu.begin());
    const auto count=entries.size();
    Exl3ResourceInventory::Requirement retained;
    retained.configuration=configuration;retained.units=plan.retained_after_capture;
    Exl3ResourceInventory produced;
    authority.allocate_startup_peak_resources(plan.additional_peak,retained,[&](auto accepted) {
        if(accepted!=configuration)throw std::logic_error("graph menu reservation identity");
        for(std::size_t i=0;i<count;++i) {
            const auto entry=menu[i];
            if(entry.disposition==Exl3GraphCaptureDisposition::eager_fallback)
                continue;
            auto actual=factory(entry);
            const auto units=actual.totals();
            if(units!=entry.extent.retained)
                throw std::invalid_argument("graph menu entry retained extent mismatch");
            const auto before=produced.totals();
            produced.append(actual);
            const auto after=produced.totals();
            for(unsigned domain=0;domain<units.size();++domain)
                if(after[domain]-before[domain]!=units[domain])
                    throw std::invalid_argument("graph menu reused entry allocation owner");
        }
        return produced;
    },[&]() noexcept {rollback();produced={};},std::forward<Observe>(observe));
    return produced;
}
}
