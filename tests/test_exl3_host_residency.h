#pragma once
#include "exl3_host_residency.h"

void run_host_residency_qualification(Exl3TextModel& target,const std::vector<std::int64_t>& source) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    using State=ninfer::exl3::Exl3ExactHostState;
    qualify_residency_page_probe();
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto exact=target.create_context(true);exact->prepare_continuation(8);
    int cases=0;
    for(int prefix:{63,64,65}) {
        const auto root=Request::initialize(*exact,std::span<const std::int64_t>(source.data(),prefix));
        const auto child=root->append_prompt(*exact,std::span<const std::int64_t>(source.data()+prefix,3));
        const auto separate=Request::initialize(*exact,std::span<const std::int64_t>(source.data(),prefix));
        std::vector<std::shared_ptr<const Request>> requests{root,child,separate};
        std::vector<std::shared_ptr<const State>> images{root->state(),child->state(),separate->state()};
        const auto stats=State::storage_stats(images);
        Exl3HostResidencyProbe probe;
        std::uint64_t allocations=0;
        Request::visit_host_allocations(requests,[&](const void* pointer,std::size_t bytes){probe.add(pointer,bytes);allocations+=bytes;});
        const auto measured=probe.measure();
        const auto token_bytes=Request::token_storage_stats(requests).allocated_bytes;
        require(allocations==stats.allocated_kv_bytes+stats.unique_other_payload_bytes+Request::allocated_tap_bytes(requests)+token_bytes &&
            measured.allocated_union_bytes==allocations && measured.resident_tensor_bytes<=allocations,
            "actual host tensor allocation/residency accounting");
        Exl3HostResidencyProbe payload_probe;
        Request::visit_host_allocations(requests,[&](const void* pointer,std::size_t bytes){payload_probe.add(pointer,bytes);},false);
        const auto payload=payload_probe.measure();
        require(payload.allocated_union_bytes==stats.materialized_kv_bytes+stats.unique_other_payload_bytes+Request::allocated_tap_bytes(requests)+
            Request::token_storage_stats(requests).materialized_token_records*8 &&
            payload.resident_tensor_bytes==payload.allocated_union_bytes,"actual payload residency differs from reserved capacity");
        requests.push_back(root);images.push_back(root->state());
        std::uint64_t duplicate_bytes=0;
        Request::visit_host_allocations(requests,[&](const void*,std::size_t bytes){duplicate_bytes+=bytes;});
        require(duplicate_bytes==allocations && State::storage_stats(images).logical_context_tokens==stats.logical_context_tokens+prefix,
            "alias inflated physical allocation accounting");
        const auto oracle=root->append_prompt(*exact,std::span<const std::int64_t>(source.data()+prefix,3));
        require(child->state()->same_payload(*oracle->state()) && child->same_taps(*oracle),"residency probe changed state");
        std::cout << "HOST_RESIDENCY_CASE prefix=" << prefix << " logical_tokens=" << stats.logical_context_tokens
            << " unique_kv_rows=" << stats.materialized_kv_token_rows << " tensor_allocated_bytes=" << allocations
            << " tensor_resident_bytes=" << measured.resident_tensor_bytes << " page_coverage_bytes=" << measured.page_coverage_bytes
            << " payload_bytes=" << payload.allocated_union_bytes << " payload_resident_bytes=" << payload.resident_tensor_bytes
            << " resident_page_bytes=" << measured.resident_page_bytes << " locked_tensor_bytes=" << measured.locked_tensor_bytes
            << " query_calls=" << measured.query_calls << " query_ms=" << measured.elapsed_ms << std::endl;
        ++cases;
    }
    std::cout << "HOST_RESIDENCY PASS model_cases=" << cases << " page_oracle=overlap_decommit measurement=read_only\n";
}
