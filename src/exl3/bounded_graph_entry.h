#pragma once
#include "exl3/graph_capture_requirements.h"
#include "exl3/projection_graph_binding.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>

namespace ninfer::exl3 {
struct Exl3GraphBoundResource {
    std::shared_ptr<const void> owner;
    const void* address=nullptr;
    std::size_t bytes=0;
    std::uint64_t slot=0;
};

// Allocation-free lifecycle record kept adjacent to one definition/executable
// pair. Native handles are owned by the enclosing entry; this object retains
// their declared external resources until successful handle destruction.
class Exl3BoundedGraphEntry {
public:
    static constexpr std::size_t resource_capacity=16;
    enum class Phase : std::uint8_t { empty,bound,invalidated,quarantined };
    struct Snapshot {
        Phase phase=Phase::empty;
        std::uint64_t generation=0,replay_serial=0;
        std::uintptr_t pending_stream=0;
        bool pending=false;
        std::size_t resource_count=0;
        std::uint64_t resource_bytes=0;
        Exl3ResourceInventory::Totals retained{};
        Exl3ResourceInventory::Totals temporary{};
        int first_error=0;
        std::string_view invalidation_reason;
    };

    void bind(Exl3GraphCompatibilityFingerprint fingerprint,
        std::span<const Exl3GraphBoundResource> resources,
        std::uint64_t generation,std::uint64_t reservation_configuration,
        const Exl3GraphCaptureExtent& reserved_peak) {
        if(phase_!=Phase::empty || has_pending_)
            throw std::logic_error("graph entry bind before final use");
        const auto graph_domain=static_cast<unsigned>(
            Exl3ResourceInventory::Domain::graph_count);
        if(!fingerprint.valid() || !generation || !reservation_configuration ||
           !reserved_peak.known || reserved_peak.retained[graph_domain]<2 ||
           resources.empty() || resources.size()>resource_capacity)
            throw std::invalid_argument("graph entry incomplete binding/reservation");
        for(std::size_t i=0;i<resources.size();++i) {
            const auto& resource=resources[i];
            if(!resource.owner || !resource.address || !resource.bytes)
                throw std::invalid_argument("graph entry incomplete bound resource");
            for(std::size_t j=0;j<i;++j)
                if(resources[j].slot==resource.slot && same(resources[j].owner,resource.owner))
                    throw std::invalid_argument("graph entry duplicate bound resource slot");
        }
        release_storage();
        fingerprint_=std::move(fingerprint);
        std::copy(resources.begin(),resources.end(),resources_.begin());
        resource_count_=resources.size();generation_=generation;
        reservation_configuration_=reservation_configuration;
        retained_=reserved_peak.retained;temporary_=reserved_peak.temporary;
        phase_=Phase::bound;reason_size_=0;first_error_=0;replay_serial_=0;
    }
    bool admits(const Exl3GraphCompatibilityFingerprint& fingerprint,
        std::uint64_t generation) const noexcept {
        return phase_==Phase::bound && !pending_error_ && generation==generation_ &&
            fingerprint_.matches(fingerprint);
    }
    std::uint64_t begin_replay(std::uint64_t generation,cudaStream_t stream) {
        if(phase_!=Phase::bound || generation!=generation_ || pending_error_ ||
           (has_pending_ && pending_stream_!=stream) || replay_serial_==UINT64_MAX)
            throw std::logic_error("graph entry replay owner/final use unavailable");
        pending_stream_=stream;has_pending_=true;return ++replay_serial_;
    }
    bool complete_after_drain(cudaStream_t stream,std::uint64_t replay_serial,
        int error) noexcept {
        if(!has_pending_ || stream!=pending_stream_ ||
           replay_serial!=replay_serial_ || !replay_serial)return false;
        pending_stream_=nullptr;has_pending_=false;
        if(error) {pending_error_=error;if(!first_error_)first_error_=error;}
        return !error;
    }
    void invalidate(std::string_view reason) noexcept {
        if(phase_==Phase::empty || phase_==Phase::quarantined)return;
        if(phase_==Phase::bound) {
            set_reason(reason);phase_=Phase::invalidated;
        }
    }
    void quarantine(std::string_view reason,int error) noexcept {
        if(phase_==Phase::empty)return;
        if(phase_==Phase::bound)set_reason(reason);
        if(error && !first_error_)first_error_=error;
        phase_=Phase::quarantined;
    }
    bool release_after_destroy() noexcept {
        if(phase_!=Phase::invalidated || has_pending_ || pending_error_)return false;
        release_storage();phase_=Phase::empty;return true;
    }
    bool pending() const noexcept {return has_pending_;}
    std::uint64_t replay_serial() const noexcept {return replay_serial_;}
    Snapshot snapshot() const noexcept {
        std::uint64_t resource_bytes=0;
        for(std::size_t i=0;i<resource_count_;++i) {
            const auto bytes=static_cast<std::uint64_t>(resources_[i].bytes);
            if(bytes>UINT64_MAX-resource_bytes) {
                resource_bytes=UINT64_MAX;
                break;
            }
            resource_bytes+=bytes;
        }
        return {phase_,generation_,replay_serial_,
            reinterpret_cast<std::uintptr_t>(pending_stream_),has_pending_,resource_count_,
            resource_bytes,retained_,temporary_,first_error_,
            std::string_view(reason_.data(),reason_size_)};
    }
private:
    static bool same(const std::shared_ptr<const void>& a,
        const std::shared_ptr<const void>& b) noexcept {
        return !a.owner_before(b) && !b.owner_before(a);
    }
    void set_reason(std::string_view reason) noexcept {
        if(reason_size_)return;
        const auto source=reason.empty()?std::string_view("unspecified graph invalidation"):reason;
        reason_size_=std::min(source.size(),reason_.size()-1);
        std::copy_n(source.data(),reason_size_,reason_.data());
    }
    void release_storage() noexcept {
        for(std::size_t i=0;i<resource_count_;++i)resources_[i]={};
        resource_count_=0;fingerprint_={};generation_=0;
        reservation_configuration_=0;retained_={};temporary_={};
        pending_stream_=nullptr;has_pending_=false;pending_error_=0;replay_serial_=0;
    }
    Exl3GraphCompatibilityFingerprint fingerprint_;
    std::array<Exl3GraphBoundResource,resource_capacity> resources_{};
    Exl3ResourceInventory::Totals retained_{},temporary_{};
    std::array<char,160> reason_{};
    std::size_t resource_count_=0,reason_size_=0;
    std::uint64_t generation_=0,reservation_configuration_=0,replay_serial_=0;
    cudaStream_t pending_stream_=nullptr;
    bool has_pending_=false;
    int pending_error_=0,first_error_=0;
    Phase phase_=Phase::empty;
};
}
