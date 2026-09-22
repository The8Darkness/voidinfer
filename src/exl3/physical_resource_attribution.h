#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>

namespace ninfer::exl3 {

// Prepared, opt-in collection surface. No production provider is installed by
// this type. Registered-host and locked-resident observations may overlap and
// therefore this record deliberately exposes no additive physical total.
class Exl3PhysicalResourceAttributionCollector {
public:
    using Clock=std::chrono::steady_clock;
    enum class Metric : std::uint8_t {
        host_allocation,
        cuda_registered_host,
        locked_resident_host,
        device_allocation,
        count
    };
    enum class Status : std::uint8_t {
        complete,
        partial,
        unavailable,
        provider_error,
        stale,
        observed_below_accounted
    };
    static constexpr std::size_t metric_count=static_cast<std::size_t>(Metric::count);
    static constexpr std::size_t maximum_samples=16;

    struct Accounted {
        std::array<std::uint64_t,metric_count> current{};
        std::array<std::uint64_t,metric_count> peak{};

        void validate() const {
            for(std::size_t i=0;i<metric_count;++i)if(current[i]>peak[i])
                throw std::invalid_argument("physical attribution current exceeds accounted peak");
        }
    };
    struct Observation {
        std::array<std::optional<std::uint64_t>,metric_count> current{};
        Clock::time_point observed_at{};
        int error=0;
    };
    using Provider=std::function<Observation()>;

    struct Sample {
        std::uint64_t sequence=0;
        Accounted accounted;
        std::array<std::optional<std::uint64_t>,metric_count> measured_current{};
        std::array<std::optional<std::uint64_t>,metric_count> measured_peak{};
        // Present only when the measured extent covers the exact accounted
        // extent. Absence means unknown, never an inferred zero.
        std::array<std::optional<std::uint64_t>,metric_count> unaccounted_gap{};
        Status status=Status::unavailable;
        int provider_error=0;
    };

    const Sample& collect(const Accounted& accounted,const Provider& provider,
                          Clock::time_point now,Clock::duration maximum_age) {
        accounted.validate();
        if(!provider)throw std::invalid_argument("physical attribution provider missing");
        if(size_==maximum_samples)
            throw std::length_error("physical attribution sample capacity exceeded");
        Sample sample;sample.sequence=next_sequence_++;sample.accounted=accounted;
        Observation observation;
        try { observation=provider(); }
        catch(...) {sample.status=Status::provider_error;sample.provider_error=-1;
            return append(sample);}
        if(observation.error) {
            sample.status=Status::provider_error;sample.provider_error=observation.error;
            return append(sample);
        }
        if(observation.observed_at==Clock::time_point{} ||
           maximum_age<Clock::duration::zero() || now<observation.observed_at ||
           now-observation.observed_at>maximum_age) {
            sample.status=Status::stale;return append(sample);
        }
        bool any=false,all=true,below=false;
        for(std::size_t i=0;i<metric_count;++i) {
            const auto measured=observation.current[i];
            if(!measured) {all=false;continue;}
            any=true;sample.measured_current[i]=measured;
            if(!measured_peaks_[i] || *measured>*measured_peaks_[i])
                measured_peaks_[i]=measured;
            sample.measured_peak[i]=measured_peaks_[i];
            if(*measured>=accounted.current[i])
                sample.unaccounted_gap[i]=*measured-accounted.current[i];
            else below=true;
        }
        if(below)sample.status=Status::observed_below_accounted;
        else if(all)sample.status=Status::complete;
        else sample.status=any?Status::partial:Status::unavailable;
        return append(sample);
    }

    [[nodiscard]] std::size_t size() const noexcept{return size_;}
    [[nodiscard]] const Sample& operator[](std::size_t index) const {
        if(index>=size_)throw std::out_of_range("physical attribution sample index");
        return samples_[index];
    }

private:
    const Sample& append(const Sample& sample) {
        samples_[size_]=sample;return samples_[size_++];
    }
    std::array<Sample,maximum_samples> samples_{};
    std::array<std::optional<std::uint64_t>,metric_count> measured_peaks_{};
    std::size_t size_=0;
    std::uint64_t next_sequence_=1;
};

} // namespace ninfer::exl3
