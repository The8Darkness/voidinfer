#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace ninfer::exl3::test {

enum class FutureContextEvidence : std::uint8_t {
    native_request,
    aggregate_admission,
    cached_history,
    semantic_stop,
    model_quality,
    unsupported_extent,
};

struct FutureContextQualificationRow {
    std::string_view id;
    FutureContextEvidence evidence = FutureContextEvidence::unsupported_extent;
    std::uint32_t maximum_context = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t output_budget = 0;
    std::uint32_t concurrency = 0;
    std::uint64_t aggregate_admitted_tokens = 0;
    std::uint32_t cached_history_tokens = 0;
    std::uint64_t minimum_exact_host_bytes = 0;
    std::string_view required_stop;
    bool executable = false;
    bool exact_state_required = false;
    bool independent_reference_required = false;
    // Future observations are deliberately absent, not fabricated zeros.
    std::optional<double> observed_latency_ms;
    std::optional<std::uint64_t> observed_peak_bytes;
    std::optional<double> observed_quality_score;
};

inline constexpr std::array future_context_qualification_rows{
    FutureContextQualificationRow{
        "native32_exact_capacity",FutureContextEvidence::native_request,
        32768,32768,0,1,32768,0,2ULL<<30,"context_capacity",true,true,false},
    FutureContextQualificationRow{
        "native64_near_boundary",FutureContextEvidence::native_request,
        65536,65520,16,1,65536,0,4ULL<<30,"explicit_stop_or_output_limit",true,true,false},
    FutureContextQualificationRow{
        "ordinary_c2_is_not_native64",FutureContextEvidence::aggregate_admission,
        32768,16384,16,2,65536,0,0,"per_request_stop",true,true,false},
    FutureContextQualificationRow{
        "cached_4k_cold_parity",FutureContextEvidence::cached_history,
        32768,8192,16,1,32768,4096,0,"same_as_cold",true,true,false},
    FutureContextQualificationRow{
        "cached_16k_cold_parity",FutureContextEvidence::cached_history,
        32768,24576,16,1,32768,16384,0,"same_as_cold",true,true,false},
    FutureContextQualificationRow{
        "native64_capacity_stop",FutureContextEvidence::semantic_stop,
        65536,65536,0,1,65536,0,4ULL<<30,"context_capacity",true,true,false},
    FutureContextQualificationRow{
        "native64_heldout_quality",FutureContextEvidence::model_quality,
        65536,32768,128,1,65536,0,4ULL<<30,"fixture_declared",true,true,true},
    FutureContextQualificationRow{
        "candidate128_static_only",FutureContextEvidence::unsupported_extent,
        131072,0,0,1,0,0,8ULL<<30,"execution_refused",false,false,false},
    FutureContextQualificationRow{
        "million_token_refused",FutureContextEvidence::unsupported_extent,
        1000000,0,0,1,0,0,0,"extent_refused",false,false,false},
};

inline void check_future_context_qualification_matrix_source() {
    for(std::size_t i=0;i<future_context_qualification_rows.size();++i) {
        const auto& row=future_context_qualification_rows[i];
        if(row.id.empty() || !row.concurrency || row.required_stop.empty())
            throw std::logic_error("future context matrix row identity");
        for(std::size_t j=0;j<i;++j)
            if(row.id==future_context_qualification_rows[j].id)
                throw std::logic_error("future context matrix duplicate row");
        if(row.observed_latency_ms || row.observed_peak_bytes || row.observed_quality_score)
            throw std::logic_error("unexecuted future context matrix contains observations");
        if(row.executable) {
            if((row.maximum_context!=32768 && row.maximum_context!=65536) ||
               row.prompt_tokens>row.maximum_context ||
               row.aggregate_admitted_tokens!=
                    static_cast<std::uint64_t>(row.maximum_context)*row.concurrency)
                throw std::logic_error("future supported context extent/accounting");
        } else if(row.evidence!=FutureContextEvidence::unsupported_extent ||
                  row.prompt_tokens || row.aggregate_admitted_tokens) {
            throw std::logic_error("future unsupported extent advertised work");
        }
        switch(row.evidence) {
        case FutureContextEvidence::native_request:
            if(row.concurrency!=1 || !row.exact_state_required || row.cached_history_tokens)
                throw std::logic_error("native request evidence conflated another metric");
            break;
        case FutureContextEvidence::aggregate_admission:
            if(row.concurrency<2 || row.maximum_context!=32768 ||
               row.aggregate_admitted_tokens<=row.maximum_context)
                throw std::logic_error("aggregate admission impersonates native extent");
            break;
        case FutureContextEvidence::cached_history:
            if((row.cached_history_tokens!=4096 && row.cached_history_tokens!=16384) ||
               row.cached_history_tokens>=row.prompt_tokens || !row.exact_state_required)
                throw std::logic_error("cached history evidence lacks cold exact parity");
            break;
        case FutureContextEvidence::semantic_stop:
            if(row.output_budget || row.required_stop!="context_capacity")
                throw std::logic_error("semantic stop continued diagnostic work");
            break;
        case FutureContextEvidence::model_quality:
            if(!row.independent_reference_required || row.output_budget<128)
                throw std::logic_error("model quality row lacks independent heldout evidence");
            break;
        case FutureContextEvidence::unsupported_extent:
            if(row.executable || (row.maximum_context!=131072 && row.maximum_context!=1000000))
                throw std::logic_error("impossible context row was promoted");
            break;
        }
    }
}

} // namespace ninfer::exl3::test
