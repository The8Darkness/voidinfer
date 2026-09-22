#include "ninfer/engine.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t filetime_ticks(const FILETIME& value) {
    ULARGE_INTEGER ticks{};
    ticks.LowPart  = value.dwLowDateTime;
    ticks.HighPart = value.dwHighDateTime;
    return ticks.QuadPart;
}

std::uint64_t process_cpu_ticks() {
    FILETIME creation{}, exit{}, kernel{}, user{};
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) {
        return 0;
    }
    return filetime_ticks(kernel) + filetime_ticks(user);
}

void print_load_probe(const ninfer::LoadSummary& load, double wall_seconds) {
    std::cout << "load_probe wall_seconds=" << wall_seconds
              << " load_seconds=" << load.load_seconds
              << " upload_seconds=" << load.upload_seconds
              << " reader_open_map_seconds=" << load.reader_open_map_seconds
              << " directory_parse_seconds=" << load.directory_parse_seconds
              << " directory_validate_seconds=" << load.directory_validate_seconds
              << " direct_read_seconds=" << load.direct_read_seconds
              << " direct_read_bytes=" << load.direct_read_bytes
              << " direct_read_requests=" << load.direct_read_requests
              << " direct_read_min_bytes=" << load.direct_read_min_bytes
              << " direct_read_max_bytes=" << load.direct_read_max_bytes
              << " direct_read_max_outstanding=" << load.direct_read_max_outstanding
              << " dual_source=" << (load.dual_source ? "true" : "false")
              << " dual_max_parallel_reads=" << load.dual_max_parallel_reads
              << " dual_direct_read_wall_seconds=" << load.dual_direct_read_wall_seconds
              << " secondary_direct_read_seconds=" << load.secondary_direct_read_seconds
              << " secondary_direct_read_bytes=" << load.secondary_direct_read_bytes
              << " secondary_direct_read_requests=" << load.secondary_direct_read_requests
              << " secondary_direct_read_min_bytes=" << load.secondary_direct_read_min_bytes
              << " secondary_direct_read_max_bytes=" << load.secondary_direct_read_max_bytes
              << " secondary_direct_read_max_outstanding="
              << load.secondary_direct_read_max_outstanding
              << " device_allocation_seconds=" << load.device_allocation_seconds
              << " host_staging_allocation_seconds=" << load.host_staging_allocation_seconds
              << " host_resource_copy_seconds=" << load.host_resource_copy_seconds
              << " h2d_stream_seconds=" << load.h2d_stream_seconds
              << " h2d_active_seconds=" << load.h2d_active_seconds
              << " materialization_sync_seconds=" << load.materialization_sync_seconds
              << " tensor_binding_seconds=" << load.tensor_binding_seconds
              << " planner_seconds=" << load.planner_seconds
              << " instance_seconds=" << load.instance_seconds
              << " startup_sync_seconds=" << load.startup_sync_seconds
              << " artifact_bytes_read=" << load.artifact_bytes_read
              << " host_to_device_bytes=" << load.host_to_device_bytes
              << " peak_staging_bytes=" << load.peak_staging_bytes
              << " tensors=" << load.tensor_count
              << " resources=" << load.resource_count << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 6) {
        std::cerr << "usage: " << argv[0]
                  << " <model.ninfer> [max_context] [secondary.ninfer] "
                     "[default|static|dynamic|single] [--smoke]\n";
        return 2;
    }
    try {
        const bool smoke = argc >= 3 && std::string(argv[argc - 1]) == "--smoke";
        const int base_argc = argc - (smoke ? 1 : 0);
        if (base_argc != 2 && base_argc != 3 && base_argc != 5) {
            throw std::invalid_argument(
                "expected model, optional max_context, or max_context + secondary + scheduler");
        }
        const std::uint32_t max_context =
            base_argc == 3 || base_argc == 5
                ? static_cast<std::uint32_t>(std::stoul(argv[2]))
                : 512U;
        if (max_context == 0) { throw std::invalid_argument("max_context must be positive"); }

        ninfer::EngineOptions options;
        options.artifact_path   = argv[1];
        if (base_argc == 5) {
            options.secondary_artifact_path = argv[3];
            const std::string mode = argv[4];
            if (mode == "static") {
                options.artifact_read_mode = ninfer::ArtifactReadMode::DualStaticAlternating;
            } else if (mode == "dynamic") {
                options.artifact_read_mode = ninfer::ArtifactReadMode::DualDynamic;
            } else if (mode == "single") {
                options.disable_dual_artifact_loading = true;
            } else if (mode != "default" && mode != "auto") {
                throw std::invalid_argument(
                    "scheduler must be default, static, dynamic, or single");
            }
        }
        options.max_context     = max_context;
        options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
        options.prefill_chunk   = std::min<std::uint32_t>(max_context, 1024U);
        options.max_concurrency = 1;
        options.speculative.backend = ninfer::SpeculativeBackend::None;
        options.enable_vision       = false;
        options.use_cuda_graph      = false;
        options.context_cache.enabled = false;
        options.hierarchical_vericache.enabled = false;

        const auto started    = Clock::now();
        const auto cpu_start = process_cpu_ticks();
        ninfer::Engine engine(std::move(options));
        const double wall_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        const auto cpu_end        = process_cpu_ticks();
        const double cpu_seconds  = static_cast<double>(cpu_end - cpu_start) / 1.0e7;
        print_load_probe(engine.load_summary(), wall_seconds);
        const auto mode_name = [](ninfer::ArtifactReadMode mode) {
            switch (mode) {
            case ninfer::ArtifactReadMode::Single: return "single";
            case ninfer::ArtifactReadMode::DualStaticAlternating: return "dual-static";
            case ninfer::ArtifactReadMode::DualDynamic: return "dual-dynamic";
            }
            return "unknown";
        };
        std::cout << "load_probe artifact_read_mode=" << mode_name(engine.options().artifact_read_mode)
                  << " primary_path=" << engine.options().artifact_path.string()
                  << " secondary_path=" << engine.options().secondary_artifact_path.string() << '\n';
        std::cout << "load_probe process_cpu_seconds=" << cpu_seconds
                  << " process_cpu_utilization_pct="
                  << (wall_seconds > 0.0 ? 100.0 * cpu_seconds / wall_seconds : 0.0) << '\n';
        if (smoke) {
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = 1;
            request.execution.sampling.temperature    = 0.0F;
            request.execution.sampling.top_k          = 1;
            request.execution.allow_prefix_reuse      = false;
            request.stop.include_model_defaults       = false;
            const std::vector<ninfer::TokenId> prompt{
                248045, 846, 198, 5834, 248046, 198, 198, 198};
            const auto result = engine.generate(engine.prepare_tokens(prompt, false), request);
            if (result.generated_token_ids.size() != 1) {
                throw std::runtime_error("deterministic smoke did not produce one token");
            }
            std::cout << "load_probe smoke_generated_token="
                      << result.generated_token_ids.front() << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "load_probe error: " << error.what() << '\n';
        return 1;
    }
}
