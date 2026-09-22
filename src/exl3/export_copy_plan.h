#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::exl3 {

enum class Exl3ExportPrecision : std::uint8_t { fp16,fp32 };

// Fixed descriptor storage only. A merge is legal when both physical ranges
// are byte-adjacent, the represented precision is identical, and neither
// logical range declares padding. No staging allocation is created.
template<std::size_t Capacity>
class Exl3ExportCopyPlan {
public:
    struct Segment {
        void* destination=nullptr;
        const void* source=nullptr;
        std::size_t bytes=0;
        std::size_t logical_ranges=0;
        Exl3ExportPrecision precision=Exl3ExportPrecision::fp16;
    };
    struct Submission {
        std::size_t attempted=0;
        std::size_t completed=0;
        std::size_t failed_segment=Capacity;
        int status=0;
        explicit operator bool() const noexcept {return status==0;}
    };

    bool add(void* destination,const void* source,std::size_t bytes,
             Exl3ExportPrecision precision,bool contains_padding=false) {
        const auto element=precision==Exl3ExportPrecision::fp16?2ULL:4ULL;
        if(!destination || !source || !bytes || bytes%element)
            throw std::invalid_argument("export copy range identity/precision");
        const auto destination_address=reinterpret_cast<std::uintptr_t>(destination);
        const auto source_address=reinterpret_cast<std::uintptr_t>(source);
        if(bytes>std::numeric_limits<std::uintptr_t>::max()-destination_address ||
           bytes>std::numeric_limits<std::uintptr_t>::max()-source_address)
            throw std::invalid_argument("export copy range overflow");
        if(count_ && !contains_padding && !last_contains_padding_) {
            auto& previous=segments_[count_-1];
            const auto previous_destination=
                reinterpret_cast<std::uintptr_t>(previous.destination);
            const auto previous_source=reinterpret_cast<std::uintptr_t>(previous.source);
            if(previous.precision==precision &&
               previous_destination+previous.bytes==destination_address &&
               previous_source+previous.bytes==source_address) {
                const auto merged_bytes=checked_add(previous.bytes,bytes);
                const auto next_logical_bytes=checked_add(logical_bytes_,bytes);
                previous.bytes=merged_bytes;
                ++previous.logical_ranges;
                ++logical_ranges_;
                logical_bytes_=next_logical_bytes;
                ++coalesced_ranges_;
                last_contains_padding_=false;
                return true;
            }
        }
        if(count_==Capacity)return false;
        const auto next_logical_bytes=checked_add(logical_bytes_,bytes);
        segments_[count_++]={destination,source,bytes,1,precision};
        ++logical_ranges_;
        logical_bytes_=next_logical_bytes;
        last_contains_padding_=contains_padding;
        return true;
    }

    template<class Submit>
    Submission submit(Submit&& submit_one) const noexcept {
        Submission result;
        for(std::size_t index=0;index<count_;++index) {
            ++result.attempted;
            const auto status=static_cast<int>(submit_one(segments_[index]));
            if(status) {
                result.status=status;
                result.failed_segment=index;
                return result;
            }
            ++result.completed;
        }
        return result;
    }
    std::size_t size() const noexcept {return count_;}
    std::size_t logical_ranges() const noexcept {return logical_ranges_;}
    std::size_t logical_bytes() const noexcept {return logical_bytes_;}
    std::size_t coalesced_ranges() const noexcept {return coalesced_ranges_;}
    const Segment& operator[](std::size_t index) const {
        if(index>=count_)throw std::out_of_range("export copy segment");
        return segments_[index];
    }
private:
    static std::size_t checked_add(std::size_t left,std::size_t right) {
        if(right>std::numeric_limits<std::size_t>::max()-left)
            throw std::overflow_error("export copy byte extent");
        return left+right;
    }
    std::array<Segment,Capacity> segments_{};
    std::size_t count_=0,logical_ranges_=0,logical_bytes_=0,coalesced_ranges_=0;
    bool last_contains_padding_=false;
};

} // namespace ninfer::exl3
