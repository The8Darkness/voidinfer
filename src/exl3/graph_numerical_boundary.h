#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

namespace ninfer::exl3 {

// Admission descriptor only. Capture setup and all host-facing work remain
// outside this boundary; the captured body may contain numerical device work.
class Exl3GraphNumericalBoundary {
public:
    enum Work : std::uint32_t {
        numerical=1u<<0,
        host_allocation=1u<<1,
        host_registration=1u<<2,
        host_export=1u<<3,
        publication=1u<<4,
        dynamic_callback=1u<<5,
        external_transfer=1u<<6,
        dynamic_extent=1u<<7
    };
    enum class Disposition : std::uint8_t { capture,eager_fallback };
    enum class Reason : std::uint8_t {
        eligible,missing_numerical_work,illegal_captured_work,missing_owner,
        missing_stream_owner,stale_external_dependency,unstable_geometry
    };
    struct Assessment {
        Disposition disposition=Disposition::eager_fallback;
        Reason reason=Reason::missing_numerical_work;
        constexpr bool eligible() const noexcept {
            return disposition==Disposition::capture;
        }
    };

    void bind(std::shared_ptr<const void> context,
        std::shared_ptr<const void> input,std::shared_ptr<const void> output,
        const void* input_address,const void* output_address,
        std::uintptr_t stream,std::shared_ptr<const void> stream_owner,
        std::uint64_t dependency_generation,
        std::uint64_t current_generation,bool fixed_geometry) noexcept {
        context_=std::move(context);input_=std::move(input);output_=std::move(output);
        stream_owner_=std::move(stream_owner);input_address_=input_address;
        output_address_=output_address;stream_=stream;
        dependency_generation_=dependency_generation;
        current_generation_=current_generation;fixed_geometry_=fixed_geometry;
    }
    void capture_work(std::uint32_t work) noexcept {captured_|=work;}
    void outside_work(std::uint32_t work) noexcept {outside_|=work;}
    Assessment assess() const noexcept {
        if(!(captured_&numerical))return {};
        constexpr auto forbidden=host_allocation|host_registration|host_export|
            publication|dynamic_callback|external_transfer|dynamic_extent;
        if(captured_&forbidden)
            return {Disposition::eager_fallback,Reason::illegal_captured_work};
        if(!context_ || !input_ || !output_ || !input_address_ || !output_address_)
            return {Disposition::eager_fallback,Reason::missing_owner};
        if(stream_ && !stream_owner_)
            return {Disposition::eager_fallback,Reason::missing_stream_owner};
        if(!dependency_generation_ || dependency_generation_!=current_generation_)
            return {Disposition::eager_fallback,Reason::stale_external_dependency};
        if(!fixed_geometry_)
            return {Disposition::eager_fallback,Reason::unstable_geometry};
        return {Disposition::capture,Reason::eligible};
    }
    std::uint32_t captured_work() const noexcept {return captured_;}
    std::uint32_t outside_work() const noexcept {return outside_;}
    static constexpr std::string_view reason(Reason value) noexcept {
        switch(value) {
        case Reason::eligible:return "eligible numerical-only graph boundary";
        case Reason::missing_numerical_work:return "graph boundary has no numerical work";
        case Reason::illegal_captured_work:return "graph boundary contains host/dynamic work";
        case Reason::missing_owner:return "graph boundary input/output owner is missing";
        case Reason::missing_stream_owner:return "graph boundary stream owner is missing";
        case Reason::stale_external_dependency:return "graph boundary dependency generation is stale";
        case Reason::unstable_geometry:return "graph boundary geometry is not fixed";
        }
        return "unknown graph boundary disposition";
    }
private:
    std::shared_ptr<const void> context_,input_,output_,stream_owner_;
    const void* input_address_=nullptr;
    const void* output_address_=nullptr;
    std::uintptr_t stream_=0;
    std::uint64_t dependency_generation_=0,current_generation_=0;
    std::uint32_t captured_=0,outside_=0;
    bool fixed_geometry_=false;
};

} // namespace ninfer::exl3
