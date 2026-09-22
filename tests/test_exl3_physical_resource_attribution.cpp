#include "exl3/physical_resource_attribution.h"

#include <iostream>
#include <stdexcept>

using Collector=ninfer::exl3::Exl3PhysicalResourceAttributionCollector;
namespace {
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> bool refuses(F&& call){try{call();}catch(...){return true;}return false;}
constexpr std::size_t metric(Collector::Metric value){return static_cast<std::size_t>(value);}
}

int main() {
    Collector collector;
    Collector::Accounted accounted;
    accounted.current={100,40,32,200};accounted.peak={120,48,40,220};
    const auto now=Collector::Clock::time_point{}+std::chrono::seconds(10);
    unsigned calls=0;
    const auto& first=collector.collect(accounted,[&] {
        ++calls;Collector::Observation result;
        result.current={110,45,std::nullopt,260};result.observed_at=now;return result;
    },now,std::chrono::seconds(1));
    need(calls==1 && first.status==Collector::Status::partial &&
            first.unaccounted_gap[metric(Collector::Metric::host_allocation)]==10 &&
            first.unaccounted_gap[metric(Collector::Metric::cuda_registered_host)]==5 &&
            !first.unaccounted_gap[metric(Collector::Metric::locked_resident_host)] &&
            first.unaccounted_gap[metric(Collector::Metric::device_allocation)]==60,
        "partial physical sample replaced unknown locked bytes or lost exact gaps");

    const auto& second=collector.collect(accounted,[&] {
        Collector::Observation result;result.current={105,44,38,250};
        result.observed_at=now;return result;
    },now,std::chrono::seconds(1));
    need(second.status==Collector::Status::complete &&
            second.measured_peak[metric(Collector::Metric::host_allocation)]==110 &&
            second.measured_peak[metric(Collector::Metric::cuda_registered_host)]==45 &&
            second.measured_peak[metric(Collector::Metric::locked_resident_host)]==38 &&
            second.measured_peak[metric(Collector::Metric::device_allocation)]==260,
        "bounded collection lost independent measured peaks");

    const auto& failed=collector.collect(accounted,[] {
        Collector::Observation result;result.error=37;return result;
    },now,std::chrono::seconds(1));
    need(failed.status==Collector::Status::provider_error && failed.provider_error==37 &&
            !failed.measured_current[metric(Collector::Metric::device_allocation)],
        "provider failure became a zero-byte physical observation");
    const auto& missing=collector.collect(accounted,[&] {
        Collector::Observation result;result.observed_at=now;return result;
    },now,std::chrono::seconds(1));
    need(missing.status==Collector::Status::unavailable &&
            !missing.unaccounted_gap[metric(Collector::Metric::device_allocation)],
        "unavailable physical metric became a zero driver gap");
    const auto& contradictory=collector.collect(accounted,[&] {
        Collector::Observation result;result.current[metric(Collector::Metric::device_allocation)]=199;
        result.observed_at=now;return result;
    },now,std::chrono::seconds(1));
    need(contradictory.status==Collector::Status::observed_below_accounted &&
            !contradictory.unaccounted_gap[metric(Collector::Metric::device_allocation)],
        "observation below accounted device bytes underflowed a driver gap");
    Collector::Accounted invalid=accounted;invalid.current[0]=121;
    need(refuses([&]{(void)collector.collect(invalid,[] {return Collector::Observation{};},
            now,std::chrono::seconds(1));}),
        "invalid accounted peak reached a provider");
    const auto& thrown=collector.collect(accounted,[]() -> Collector::Observation {
        throw std::runtime_error("synthetic provider failure");
    },now,std::chrono::seconds(1));
    need(thrown.status==Collector::Status::provider_error && thrown.provider_error==-1,
        "throwing provider escaped the telemetry boundary");
    const auto& stale=collector.collect(accounted,[&] {
        Collector::Observation result;result.current[0]=110;
        result.observed_at=now-std::chrono::seconds(2);return result;
    },now,std::chrono::seconds(1));
    need(stale.status==Collector::Status::stale && !stale.measured_current[0],
        "stale physical observation changed measured state");
    while(collector.size()<Collector::maximum_samples)
        (void)collector.collect(accounted,[&] {
            Collector::Observation result;result.observed_at=now;return result;
        },now,std::chrono::seconds(1));
    unsigned over_capacity_calls=0;
    need(refuses([&]{(void)collector.collect(accounted,[&] {
            ++over_capacity_calls;return Collector::Observation{};
        },now,std::chrono::seconds(1));}) && over_capacity_calls==0,
        "bounded collector invoked a provider after filling its record arena");
    std::cout<<"exl3_physical_resource_attribution PASS\n";
}
