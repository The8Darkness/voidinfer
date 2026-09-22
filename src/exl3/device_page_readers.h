#pragma once
#include "exl3/device_page_copy.h"
#include "exl3/device_page_attention.h"
#include <array>

namespace ninfer::exl3 {
// Fixed Engine reader pool: admit every physical lane before constructing any
// reader. Cache storage and request callback installation are separate owners.
template<std::size_t Capacity> struct Exl3DevicePageReaders {
    static_assert(Capacity>0 && Capacity<=64);
    std::array<std::shared_ptr<Exl3DevicePageCopy>,2> copies{};
    std::array<std::array<std::shared_ptr<Exl3DevicePageAttention>,Capacity>,2> attention{};
    static Exl3ResourceInventory::Requirement requirement(unsigned lanes,bool direct) {
        if(lanes<1 || lanes>2)throw std::invalid_argument("page reader lane count");
        Exl3ResourceInventory::Requirement required;required.configuration=0x5047524452;
        required.add(Exl3ResourceInventory::Domain::host_metadata,lanes,Exl3DevicePageCopy::metadata_bytes());
        if(direct)required.add(Exl3ResourceInventory::Domain::host_metadata,lanes*Capacity,
            Exl3DevicePageAttention::metadata_bytes());
        return required;
    }
    template<class Coordinator>
    static Exl3DevicePageReaders create_startup(Coordinator& authority,unsigned lanes,bool direct,
        unsigned fail_after_reader_for_test=0) {
        const auto required=requirement(lanes,direct);
        const auto count=lanes*(1+(direct?Capacity:0));
        if(fail_after_reader_for_test>count)throw std::invalid_argument("page reader construction fault index");
        Exl3DevicePageReaders result;
        authority.allocate_startup_resources(required,[&](std::uint64_t configuration) {
            if(configuration!=required.configuration)throw std::logic_error("page reader reservation identity");
            Exl3DevicePageReaders pending;Exl3ResourceInventory actual;unsigned constructed=0;
            const auto add=[&]<class Owner>(std::shared_ptr<Owner>& owner) {
                owner=make_bounded_shared<Owner>();
                actual.add({owner,0,Exl3ResourceInventory::Domain::host_metadata,Owner::metadata_bytes(),{},
                    &Owner::attach_retirement_credit});
                if(++constructed==fail_after_reader_for_test)
                    throw std::runtime_error("injected page reader construction failure");
            };
            for(unsigned lane=0;lane<lanes;++lane) {
                add(pending.copies[lane]);
                if(direct)for(auto& reader:pending.attention[lane])add(reader);
            }
            result=std::move(pending);return actual;
        },[&]() noexcept {result={};});
        return result;
    }
};
}
