#pragma once
#include "test_exl3_host_residency.h"

void run_t90_allocation_churn(
    Exl3TextModel& target, const std::vector<std::int64_t>& code,
    const std::filesystem::path& output) {
    using Request = ninfer::exl3::Exl3VeriCacheRequest;
    using Cache = ninfer::exl3::Exl3VeriCacheServingPrefixCache;
    using Identity = ninfer::exl3::Exl3VeriCacheServingIdentity;
    using Coordinator = ninfer::exl3::Exl3VeriCacheServingCoordinator;
    using Domains = ninfer::exl3::Exl3ExactAllocationDomainStats;
    constexpr std::size_t kPrompt = 4096;
    constexpr std::size_t kPrefix = 3968;
    constexpr int kOutputs = 8;
    require(target.max_context() >= 4352 && code.size() >= kPrompt,
            "T90 fixture extent");
    require(!output.empty() && !std::filesystem::exists(output),
            "T90 output must be new");
    std::filesystem::create_directories(output);
    const auto reserve_gib = std::stoull(env("NINFER_T90_RESERVE_GIB"));
    const auto cache_mib = std::stoull(env("NINFER_T90_CACHE_MIB"));
    require(reserve_gib >= 8 && cache_mib >= 1024, "T90 memory policy extent");
    const std::uint64_t reserve = reserve_gib * (1ULL << 30);
    const std::uint64_t cache_budget = cache_mib * (1ULL << 20);
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    require(GlobalMemoryStatusEx(&memory) != 0 && memory.ullAvailPhys > reserve,
            "T90 physical memory reserve");
    const auto available = [] {
        MEMORYSTATUSEX state{};
        state.dwLength = sizeof(state);
        require(GlobalMemoryStatusEx(&state) != 0, "T90 physical memory query");
        return state.ullAvailPhys;
    };
    const Identity identity{
        env("NINFER_T90_NAMESPACE"), env("NINFER_T90_ARTIFACT_ID"),
        env("NINFER_T90_TOKENIZER_ID"), env("NINFER_T90_CONFIGURATION_ID"),
        env("NINFER_T90_MODALITY_ID")};
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY", "0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
    Cache cache(Cache::Policy{4, 64, cache_budget, reserve}, identity);
    auto context = target.create_context(true);
    context->prepare_continuation(8);
    TargetGraphC2Stream stream;
    std::vector<std::int64_t> common(code.begin(), code.begin() + kPrefix);
    auto fill = cache.prepare(*context, common, kPrefix, available(), 1024);
    cuda_check(cudaDeviceSynchronize(), "T90 cache fill completion");
    require(!fill.metrics.cache_hit && fill.metrics.admitted, "T90 cache fill");
    fill.request.reset();
    auto prepared = cache.prepare(*context, code, kPrefix, available(), 1024);
    cuda_check(cudaDeviceSynchronize(), "T90 request preparation completion");
    require(prepared.metrics.cache_hit &&
                prepared.metrics.reused_prompt_tokens == kPrefix &&
                prepared.metrics.executed_prompt_tokens == kPrompt - kPrefix,
            "T90 preparation accounting");
    const auto initial = std::move(prepared.request);

    context->restore_exact_host_state(*initial->state(), stream.value);
    auto reference = initial;
    std::vector<std::int64_t> reference_tokens;
    auto reference_pending = sample_target(*context, stream.value);
    for (int ordinal = 0; ordinal < kOutputs; ++ordinal) {
        const std::array<std::int64_t,1> token{reference_pending};
        auto [updated, verified] = reference->verify(
            *context, token, {}, stream.value);
        require(verified.verification_rows == 1 &&
                    verified.committed_tokens.size() == 1 &&
                    verified.committed_tokens[0] == reference_pending,
                "T90 serial reference");
        reference_tokens.push_back(reference_pending);
        reference = std::move(updated);
        if (ordinal + 1 < kOutputs)
            reference_pending = sample_target(*context, stream.value);
    }

    Coordinator coordinator(cache, Coordinator::Policy{
        4, 1, memory.ullTotalPhys - reserve, reserve});
    const auto ticket = coordinator.admit(initial);
    auto acquired = coordinator.acquire();
    require(acquired && acquired->ticket.request_id == ticket.request_id,
            "T90 acquisition");
    auto lease = *acquired;
    context->restore_exact_host_state(*lease.root->state(), stream.value);
    auto pending = sample_target(*context, stream.value);
    std::vector<std::int64_t> published_tokens;
    std::ofstream rows(output / "rows.csv");
    require(rows.good(), "T90 evidence open");
    rows << "ordinal,token,kv_child_only,kv_parent_only,recurrent_child_only,"
            "recurrent_parent_only,convolution_child_only,convolution_parent_only,"
            "state_taps_child_only,state_taps_parent_only,logits_child_only,"
            "logits_parent_only,embedding_child_only,embedding_parent_only,"
            "request_taps_child_only,request_taps_parent_only,token_child_only,"
            "token_parent_only,total_child_only,total_parent_only,new_locked_bytes,"
            "unlocked_bytes,locked_page_bytes,inventory_ms,registry_ms,"
            "verification_rows,executed_rows,replay_rows,committed_rows,"
            "root_restores,exact\n";
    const auto domain_total = [](const Domains& value) {
        return value.kv + value.recurrent + value.convolution + value.state_taps +
            value.logits + value.embedding;
    };
    for (int ordinal = 0; ordinal < kOutputs; ++ordinal) {
        const auto parent = lease.root;
        const std::array<std::int64_t,1> token{pending};
        auto [child, verified] = parent->verify(*context, token, {}, stream.value);
        const std::array<std::shared_ptr<const ninfer::exl3::Exl3ExactHostState>,1>
            parent_states{parent->state()}, child_states{child->state()};
        const std::array<std::shared_ptr<const ninfer::exl3::Exl3ExactHostState>,2>
            union_states{parent->state(), child->state()};
        const auto parent_domain = ninfer::exl3::Exl3ExactHostState::
            allocation_domain_stats(parent_states);
        const auto child_domain = ninfer::exl3::Exl3ExactHostState::
            allocation_domain_stats(child_states);
        const auto union_domain = ninfer::exl3::Exl3ExactHostState::
            allocation_domain_stats(union_states);
        const std::array<std::shared_ptr<const Request>,1> parent_requests{parent};
        const std::array<std::shared_ptr<const Request>,1> child_requests{child};
        const std::array<std::shared_ptr<const Request>,2> union_requests{parent, child};
        const auto parent_taps = Request::allocated_tap_bytes(parent_requests);
        const auto child_taps = Request::allocated_tap_bytes(child_requests);
        const auto union_taps = Request::allocated_tap_bytes(union_requests);
        const auto parent_tokens = Request::token_storage_stats(parent_requests).allocated_bytes;
        const auto child_tokens = Request::token_storage_stats(child_requests).allocated_bytes;
        const auto union_tokens = Request::token_storage_stats(union_requests).allocated_bytes;
        const auto child_only = Domains{
            union_domain.kv - parent_domain.kv,
            union_domain.recurrent - parent_domain.recurrent,
            union_domain.convolution - parent_domain.convolution,
            union_domain.state_taps - parent_domain.state_taps,
            union_domain.logits - parent_domain.logits,
            union_domain.embedding - parent_domain.embedding};
        const auto parent_only = Domains{
            union_domain.kv - child_domain.kv,
            union_domain.recurrent - child_domain.recurrent,
            union_domain.convolution - child_domain.convolution,
            union_domain.state_taps - child_domain.state_taps,
            union_domain.logits - child_domain.logits,
            union_domain.embedding - child_domain.embedding};
        const auto tap_child_only = union_taps - parent_taps;
        const auto tap_parent_only = union_taps - child_taps;
        const auto token_child_only = union_tokens - parent_tokens;
        const auto token_parent_only = union_tokens - child_tokens;
        const auto before = coordinator.stats();
        const auto publication = coordinator.publish(lease, child, pending);
        const auto after = coordinator.stats();
        lease = publication.lease;
        published_tokens.push_back(pending);
        const auto total_child = domain_total(child_only) + tap_child_only +
            token_child_only;
        const auto total_parent = domain_total(parent_only) + tap_parent_only +
            token_parent_only;
        rows << ordinal << ',' << pending << ',' << child_only.kv << ','
            << parent_only.kv << ',' << child_only.recurrent << ','
            << parent_only.recurrent << ',' << child_only.convolution << ','
            << parent_only.convolution << ',' << child_only.state_taps << ','
            << parent_only.state_taps << ',' << child_only.logits << ','
            << parent_only.logits << ',' << child_only.embedding << ','
            << parent_only.embedding << ',' << tap_child_only << ','
            << tap_parent_only << ',' << token_child_only << ','
            << token_parent_only << ',' << total_child << ',' << total_parent << ','
            << after.resident_new_locked_bytes - before.resident_new_locked_bytes
            << ',' << after.resident_unlocked_bytes - before.resident_unlocked_bytes
            << ',' << after.locked_page_bytes << ','
            << after.resident_inventory_ms - before.resident_inventory_ms << ','
            << after.resident_registry_ms - before.resident_registry_ms << ','
            << verified.verification_rows << ',' << verified.executed_rows << ','
            << verified.replay_rows << ',' << verified.committed_tokens.size() << ','
            << verified.root_restores << ",1\n";
        if (ordinal + 1 < kOutputs)
            pending = sample_target(*context, stream.value);
    }
    require(published_tokens == reference_tokens &&
                lease.root->token_suffix() == reference->token_suffix() &&
                lease.root->same_taps(*reference) &&
                lease.root->state()->same_payload(*reference->state()),
            "T90 final oracle");
    std::vector<std::shared_ptr<const Request>> active = cache.roots();
    active.push_back(lease.root);
    Exl3HostResidencyProbe probe;
    Request::visit_host_allocations(active,
        [&](const void* data, std::size_t bytes) { probe.add(data, bytes); }, false);
    const auto resident = probe.measure();
    require(resident.resident_tensor_bytes == resident.allocated_union_bytes &&
                resident.locked_tensor_bytes == resident.allocated_union_bytes,
            "T90 resident publication");
    std::size_t free_bytes = 0, total_device = 0;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_device), "T90 device memory");
    require(free_bytes >= (1ULL << 30), "T90 device reserve");
    const auto final_stats = coordinator.stats();
    coordinator.complete(lease);
    coordinator.close();
    require(coordinator.stats().closed && coordinator.stats().active == 0 &&
                coordinator.stats().admitted == 0,
            "T90 completion/close");
    std::ofstream memory_out(output / "memory.csv");
    require(memory_out.good(), "T90 memory evidence open");
    const auto storage = cache.storage_stats();
    memory_out << "resident_payload_bytes,locked_payload_bytes,locked_page_bytes,"
                  "cache_payload_bytes,cache_identity_bytes,device_free_bytes,"
                  "physical_reserve_bytes\n";
    memory_out << resident.resident_tensor_bytes << ','
        << resident.locked_tensor_bytes << ',' << final_stats.locked_page_bytes
        << ',' << storage.payload_allocated_bytes << ','
        << storage.identity_allocated_bytes << ',' << free_bytes << ','
        << reserve << '\n';
    rows.flush(); memory_out.flush();
    require(rows.good() && memory_out.good(), "T90 evidence flush");
    std::cout << "T90_ALLOCATION_CHURN PASS publications=8 exact=1"
              << " resident_locked_bytes=" << resident.resident_tensor_bytes
              << " completions=1 cancellations=0"
              << " output=" << output.string() << std::endl;
}
