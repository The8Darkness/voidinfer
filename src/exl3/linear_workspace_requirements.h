#pragma once
#include "exl3/resource_inventory.h"
#include <cstddef>

namespace ninfer::exl3 {
struct Exl3LinearWorkspaceRequirements {
    static constexpr unsigned accumulation_splits=5;
    static constexpr std::size_t transformed_alignment=16;
    std::size_t transformed_bytes=0,accumulation_bytes=0,owned_bytes=0;
    // Validate the used extents, not spare capacity in a shared allocation.
    void require_disjoint_borrowed_views(const void* transformed,const void* accumulation) const {
        const auto checked_end=[](const void* pointer,std::size_t bytes) {
            const auto begin=reinterpret_cast<std::uintptr_t>(pointer);
            if(bytes>std::numeric_limits<std::uintptr_t>::max()-begin)
                throw std::overflow_error("EXL3 borrowed workspace address overflow");
            return begin+bytes;
        };
        const auto transform_end=transformed?checked_end(transformed,transformed_bytes):0;
        const auto accumulation_end=accumulation?checked_end(accumulation,accumulation_bytes):0;
        // Async-A reads 16 bytes from transformed rows. All admitted row widths
        // are multiples of 128 elements, preserving this base alignment.
        if((transformed && reinterpret_cast<std::uintptr_t>(transformed)%transformed_alignment) ||
            (accumulation && reinterpret_cast<std::uintptr_t>(accumulation)%alignof(float)))
            throw std::invalid_argument("EXL3 borrowed workspace storage alignment");
        if(transformed && accumulation && reinterpret_cast<std::uintptr_t>(transformed)<accumulation_end &&
            reinterpret_cast<std::uintptr_t>(accumulation)<transform_end)
            throw std::invalid_argument("EXL3 borrowed transform/accumulation overlap");
    }
    static std::size_t append_owned_bytes(std::size_t current,std::size_t added) {
        if(!added)throw std::invalid_argument("EXL3 workspace allocation extent missing");
        if(added>std::numeric_limits<std::size_t>::max()-current)
            throw std::overflow_error("EXL3 workspace allocation total overflow");
        return current+added;
    }
    // Factory returns a unique allocation owner exposing its device pointer as
    // ptr. Metadata growth precedes physical acquisition; publication is last.
    template<class Owners,class Factory>
    static void allocate_owned(Owners& owners,std::size_t& total,void** destination,
        std::size_t bytes,Factory&& factory) {
        if(!destination)throw std::invalid_argument("context allocation destination missing");
        const auto next_total=append_owned_bytes(total,bytes);
        if(owners.size()==owners.capacity()) {
            const auto maximum=owners.max_size(),current=owners.capacity();
            if(current==maximum)throw std::length_error("context allocation owner capacity exhausted");
            owners.reserve(current==0?std::size_t{1}:current>maximum-current?maximum:current*2);
        }
        auto owner=std::forward<Factory>(factory)();
        if(!owner || !owner->ptr || owner->bytes!=bytes)
            throw std::runtime_error("context allocation factory storage/extent mismatch");
        const auto pointer=owner->ptr;
        owners.push_back(std::move(owner));
        *destination=pointer;total=next_total;
    }
    static Exl3LinearWorkspaceRequirements derive(int input,int output,int rows,
        bool borrowed_transform=false,bool borrowed_accumulation=false) {
        if(input<=0 || output<=0 || input%128 || output%128 || rows<=0)
            throw std::invalid_argument("EXL3 workspace requirement dimensions");
        const auto extent=[](std::uint64_t count,std::uint64_t width) {
            if(count>std::numeric_limits<std::size_t>::max()/width)
                throw std::overflow_error("EXL3 workspace requirement extent overflow");
            return static_cast<std::size_t>(count*width);
        };
        Exl3LinearWorkspaceRequirements result;
        result.transformed_bytes=extent(rows,std::uint64_t(input)*sizeof(std::uint16_t));
        result.accumulation_bytes=extent(rows,std::uint64_t(output)*accumulation_splits*sizeof(float));
        const auto transformed=borrowed_transform?0:result.transformed_bytes;
        const auto accumulation=borrowed_accumulation?0:result.accumulation_bytes;
        if(accumulation>std::numeric_limits<std::size_t>::max()-transformed)
            throw std::overflow_error("EXL3 workspace requirement total overflow");
        result.owned_bytes=transformed+accumulation;
        return result;
    }
};
}
