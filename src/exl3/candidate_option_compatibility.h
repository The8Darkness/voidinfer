#pragma once

#include <cstdint>
#include <stdexcept>

namespace ninfer::exl3 {

// Bounded manifest for options introduced by the EXL3 campaign. It is consumed
// once by actual Engine construction after textual option validation and before
// model/context numerical construction. It deliberately does not become a
// second feature registry.
struct Exl3CandidateOptionCompatibility {
    unsigned physical_lanes=1;
    std::uint32_t max_context=0;
    std::uint32_t device_prefix_rows=0;
    bool exact_host_kv=false,oscar_l0_only=false;
    bool native64k=false,candidate128k=false,native_continuation16=false;
    bool device_prefix=false,shared_prefix=false,registered_upload=false;
    bool shared_pages=false,shared_page_attention=false,attention_staging=false;
    bool shared_target_q=false,shared_target_kv=false,shared_target_o=false;
    bool shared_target_gateup=false,shared_target_down=false,shared_head=false;
    bool shared_gather=false,target_stream_reduction=false;
    bool shared_draft_q=false,shared_draft_kv=false,shared_draft_o=false;
    bool shared_draft_down=false,shared_draft_gateup=false;
    bool device_greedy=false,batched_greedy=false;
};

inline void require_candidate_option_compatibility(
    const Exl3CandidateOptionCompatibility& value) {
    if((value.physical_lanes!=1 && value.physical_lanes!=2) || !value.exact_host_kv ||
       value.oscar_l0_only)
        throw std::invalid_argument("EXL3 candidates require ordinary exact-host C1/C2");
    if(value.candidate128k)
        throw std::invalid_argument("EXL3 128K candidate is static-only");
    if(value.native64k && (value.max_context!=65536 || value.physical_lanes!=1))
        throw std::invalid_argument("EXL3 native64K requires physical C1 and exact extent");
    if(value.native_continuation16)
        throw std::invalid_argument("EXL3 native16 is not an exact Engine route");
    if(value.device_prefix_rows!=0 && value.device_prefix_rows!=4096 &&
       value.device_prefix_rows!=16384)
        throw std::invalid_argument("EXL3 represented prefix row menu is 4096/16384");
    if(value.device_prefix_rows && !value.device_prefix)
        throw std::invalid_argument("EXL3 represented prefix rows require represented prefix storage");
    if(value.device_prefix_rows==16384 && value.physical_lanes!=1 && !value.shared_pages)
        throw std::invalid_argument("EXL3 represented prefix 16K requires C1");

    const bool any_target=value.shared_target_q || value.shared_target_kv ||
        value.shared_target_o || value.shared_target_gateup || value.shared_target_down ||
        value.shared_head || value.shared_gather;
    const bool any_draft=value.shared_draft_q || value.shared_draft_kv ||
        value.shared_draft_o || value.shared_draft_down || value.shared_draft_gateup;
    if((any_target || any_draft || value.shared_prefix || value.batched_greedy) &&
       value.physical_lanes!=2)
        throw std::invalid_argument("EXL3 paired candidate requires physical C2");
    if((value.shared_target_kv || value.shared_target_o || value.shared_target_gateup ||
        value.shared_target_down || value.shared_head) && !value.shared_target_q)
        throw std::invalid_argument("EXL3 shared target family requires shared target Q owner");
    if((value.shared_draft_kv || value.shared_draft_o || value.shared_draft_down ||
        value.shared_draft_gateup) && !value.shared_draft_q)
        throw std::invalid_argument("EXL3 shared draft family requires shared draft Q owner");
    if(value.shared_gather && !value.shared_target_gateup && !value.shared_draft_gateup)
        throw std::invalid_argument("EXL3 gather reuse requires a shared gate/up consumer");
    if(value.shared_target_q && value.target_stream_reduction)
        throw std::invalid_argument("EXL3 shared target and stream-reduction routes are exclusive");
    if(value.shared_prefix && !value.device_prefix)
        throw std::invalid_argument("EXL3 shared prefix requires represented prefix storage");
    if(value.shared_page_attention && !value.shared_pages)
        throw std::invalid_argument("EXL3 shared-page attention requires shared pages");
    if((value.shared_pages && value.shared_prefix) ||
       (value.attention_staging && (value.shared_pages || value.shared_prefix)))
        throw std::invalid_argument("EXL3 attention storage candidates are mutually exclusive");
    if(value.batched_greedy && !value.device_greedy)
        throw std::invalid_argument("EXL3 batched greedy requires device greedy packets");
}

}
