#pragma once
#include "exl3/resource_inventory.h"
#include <chrono>
#include <functional>
#include <charconv>
#include <string_view>

namespace ninfer::exl3 {
// Startup configuration only. Zero is an actual zero-byte cap; absence alone
// preserves the unlimited policy. This bounds inventoried host metadata, not
// authoritative host payload, OS working-set quota or CUDA registration.
inline std::uint64_t exl3_host_metadata_limit(const char* configured) {
    if(!configured)return std::numeric_limits<std::uint64_t>::max();
    const std::string_view text(configured);std::uint64_t value=0;
    const auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
    if(text.empty() || parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size())
        throw std::invalid_argument("Engine host metadata limit requires unsigned bytes");
    return value;
}
// Availability is an external observation, never an allocation owner. Providers
// report errors explicitly; fabricated zero/free values cannot stand in for a
// failed query. The authority still accounts for committed and promised bytes.
struct Exl3DeviceAvailability {
    using Clock=std::chrono::steady_clock;
    std::uint64_t total=0,available=0;
    Clock::time_point observed_at{};
    int error=0;
    using Provider=std::function<Exl3DeviceAvailability()>;
    struct Attribution {
        std::uint64_t inventoried_bytes=0;
        std::uint64_t observed_used_bytes=0;
        // This is deliberately an unknown remainder, never a leak/cache label.
        std::uint64_t driver_unknown_bytes=0;
    };
    static Exl3DeviceAvailability read(const Provider& provider) {
        if(!provider)throw std::invalid_argument("device availability provider missing");
        auto value=provider();
        if(value.error)throw std::runtime_error("device availability provider failed");
        return value;
    }
    std::uint64_t limit(std::uint64_t owned,std::uint64_t reserve,
        Clock::time_point now,Clock::duration maximum_age) const {
        if(error || observed_at==Clock::time_point{} || maximum_age<Clock::duration::zero() ||
           now<observed_at || now-observed_at>maximum_age)
            throw std::invalid_argument("device availability missing/stale observation");
        return Exl3ResourceInventory::device_limit(owned,total,available,reserve);
    }
    Attribution attribution(std::uint64_t owned,Clock::time_point now,
        Clock::duration maximum_age) const {
        (void)limit(owned,0,now,maximum_age);
        const auto used=total-available;
        return {owned,used,used-owned};
    }
};
}
