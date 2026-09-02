#include "targets/registry.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "runtime/engine/kv_capacity.h"
#include "runtime/engine/context_cost.h"

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace ninfer::targets {
namespace {

using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
    const bool dual = options.artifact_read_mode != ArtifactReadMode::Single;
    if (dual != !options.secondary_artifact_path.empty()) {
        throw std::invalid_argument(
            "dual artifact loading requires exactly one secondary .ninfer path");
    }
    if (!options.secondary_artifact_path.empty() &&
        options.secondary_artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("secondary artifact must have a .ninfer extension");
    }
}

void validate_equivalent_artifacts(const artifact::Reader& primary,
                                   const artifact::Reader& secondary) {
    if (primary.file_bytes() != secondary.file_bytes()) {
        throw std::invalid_argument("dual artifact file sizes differ");
    }
    if (primary.identity() != secondary.identity()) {
        throw std::invalid_argument("dual artifact identities differ");
    }
    const auto& left  = primary.objects();
    const auto& right = secondary.objects();
    if (left.size() != right.size()) {
        throw std::invalid_argument("dual artifact directory object counts differ");
    }
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (left[i].index() != right[i].index() || artifact::object_name(left[i]) !=
                                                       artifact::object_name(right[i]) ||
            artifact::object_offset(left[i]) != artifact::object_offset(right[i]) ||
            artifact::object_bytes(left[i]) != artifact::object_bytes(right[i])) {
            throw std::invalid_argument("dual artifact directory differs at object " +
                                        std::to_string(i));
        }
        if (const auto* a = std::get_if<artifact::TensorDescriptor>(&left[i])) {
            const auto* b = std::get_if<artifact::TensorDescriptor>(&right[i]);
            if (b == nullptr || a->shape != b->shape || a->format != b->format ||
                a->layout != b->layout) {
                throw std::invalid_argument("dual artifact tensor descriptor differs at object " +
                                            std::to_string(i));
            }
        } else {
            const auto* left_resource  = std::get_if<artifact::ResourceDescriptor>(&left[i]);
            const auto* right_resource = std::get_if<artifact::ResourceDescriptor>(&right[i]);
            if (right_resource == nullptr || left_resource->encoding != right_resource->encoding) {
                throw std::invalid_argument(
                    "dual artifact resource descriptor differs at object " + std::to_string(i));
            }
        }
    }
}

artifact::LoadProgress artifact_progress(const LoadProgress& progress) {
    return artifact::LoadProgress{.callback = progress.callback};
}

std::size_t runtime_bytes_after_planned_weights(std::uint64_t weight_bytes) {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (weight_bytes > free_bytes) {
        throw std::invalid_argument("model weights require " + std::to_string(weight_bytes) +
                                    " bytes of device memory, but only " +
                                    std::to_string(free_bytes) +
                                    " bytes are free before loading weights");
    }
    return free_bytes - static_cast<std::size_t>(weight_bytes);
}

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

template <class Target, class Loaded, class Instance>
ConstructedTarget construct_registered(const EngineOptions& options, DeviceContext& device,
                                       artifact::Reader& reader, artifact::Reader* secondary,
                                       Clock::time_point load_start,
                                       std::string_view target_key) {
    const auto& identity                          = reader.identity();
    const auto weights_profile                    = Target::resolve_weights(identity);
    const ModelSamplingDefaults sampling_defaults = Target::sampling_defaults(identity.model_id);
    const runtime::ContextCostIdentity context_cost_identity{
        .hardware_class = runtime::context_cost_hardware_class(
            device.props.name, device.props.major, device.props.minor),
        .model_id   = identity.model_id,
        .weights_id = identity.weights_id,
    };
    runtime::ResolvedContextMachineCost context_cost = runtime::resolve_context_machine_cost(
        context_cost_identity, options.context_cost.preset_path);

    const auto planner_start = Clock::now();
    artifact::Binder binder(reader);
    auto load_plan        = Target::plan_load(binder, options, weights_profile);
    auto sequence_planner = Target::make_sequence_planner(device, options, weights_profile);
    const runtime::SequenceCapacityCurve curve = sequence_planner.capacity_curve();
    const std::size_t preflight_runtime_bytes =
        runtime_bytes_after_planned_weights(load_plan.materialization().device_capacity_bytes);
    (void)runtime::resolve_kv_capacity(options.kv_capacity, curve, preflight_runtime_bytes);
    const double planner_seconds =
        std::chrono::duration<double>(Clock::now() - planner_start).count();

    auto progress     = artifact_progress(options.load_progress);
    auto materialized = secondary == nullptr
                            ? artifact::materialize(reader, load_plan.materialization(), device,
                                                    progress.callback ? &progress : nullptr)
                            : artifact::materialize_dual(
                                  reader, *secondary, load_plan.materialization(), device,
                                  options.artifact_read_mode,
                                  progress.callback ? &progress : nullptr);
    const artifact::MaterializationStats stats = materialized.stats();

    const auto model_construct_start = Clock::now();
    auto model = Target::construct_loaded_model(std::move(load_plan), std::move(materialized));
    const double model_construct_seconds =
        std::chrono::duration<double>(Clock::now() - model_construct_start).count();
    const auto pre_instance_sync_start = Clock::now();
    device.synchronize();
    const double pre_instance_sync_seconds =
        std::chrono::duration<double>(Clock::now() - pre_instance_sync_start).count();
    runtime::KvCapacityResolution capacity_resolution =
        runtime::resolve_kv_capacity(options.kv_capacity, curve, current_free_device_bytes());
    auto sequence_plan = std::move(sequence_planner).finalize(capacity_resolution.main_page_groups);
    if (sequence_plan.device_reservation_bytes() != capacity_resolution.runtime_reservation_bytes ||
        sequence_plan.kv_capacity() != capacity_resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized target plan");
    }
    const auto binding_start = Clock::now();
    auto loaded   = std::make_unique<Loaded>(std::move(model), options);
    const double binding_seconds =
        std::chrono::duration<double>(Clock::now() - binding_start).count();
    const auto instance_start = Clock::now();
    auto instance = std::make_unique<Instance>(std::move(loaded), capacity_resolution,
                                               std::move(sequence_plan), device);
    const double instance_seconds =
        std::chrono::duration<double>(Clock::now() - instance_start).count();
    const auto startup_sync_start = Clock::now();
    device.synchronize();
    const double startup_sync_seconds =
        std::chrono::duration<double>(Clock::now() - startup_sync_start).count();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();

    const auto& reader_diagnostics = reader.diagnostics();
    const auto& read_diagnostics   = reader.direct_read_diagnostics();
    LoadSummary summary;
    summary.target               = std::string(target_key);
    summary.model_id             = identity.model_id;
    summary.weights_id           = identity.weights_id;
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - load_start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.file_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.tensor_count         = stats.tensor_count;
    summary.resource_count       = stats.resource_count;
    summary.reader_open_map_seconds          = reader_diagnostics.file_open_map_seconds;
    summary.directory_parse_seconds          = reader_diagnostics.directory_parse_seconds;
    summary.directory_validate_seconds       = reader_diagnostics.directory_validate_seconds;
    summary.direct_read_seconds              = read_diagnostics.elapsed_seconds;
    summary.direct_read_requests             = read_diagnostics.request_count;
    summary.direct_read_bytes                = read_diagnostics.bytes_read;
    summary.direct_read_min_bytes            = read_diagnostics.min_request_bytes;
    summary.direct_read_max_bytes            = read_diagnostics.max_request_bytes;
    summary.direct_read_max_outstanding      = read_diagnostics.max_outstanding;
    summary.device_allocation_seconds        = stats.device_allocation_seconds;
    summary.host_staging_allocation_seconds  = stats.host_staging_allocation_seconds;
    summary.host_resource_copy_seconds       = stats.host_resource_copy_seconds;
    summary.h2d_stream_seconds               = stats.h2d_stream_seconds;
    summary.h2d_active_seconds               = stats.h2d_active_seconds;
    summary.materialization_sync_seconds    = stats.synchronization_seconds;
    summary.tensor_binding_seconds           = model_construct_seconds + binding_seconds;
    summary.planner_seconds                  = planner_seconds;
    summary.instance_seconds                 = instance_seconds;
    summary.startup_sync_seconds             = pre_instance_sync_seconds + startup_sync_seconds;
    summary.dual_source                      = secondary != nullptr;
    summary.dual_max_parallel_reads          = stats.dual_max_parallel_reads;
    summary.dual_direct_read_wall_seconds    = stats.dual_direct_read_wall_seconds;
    if (secondary != nullptr) {
        const auto& secondary_reads = secondary->direct_read_diagnostics();
        summary.secondary_direct_read_bytes = secondary_reads.bytes_read;
        summary.secondary_direct_read_requests = secondary_reads.request_count;
        summary.secondary_direct_read_min_bytes = secondary_reads.min_request_bytes;
        summary.secondary_direct_read_max_bytes = secondary_reads.max_request_bytes;
        summary.secondary_direct_read_max_outstanding = secondary_reads.max_outstanding;
        summary.secondary_direct_read_seconds = secondary_reads.elapsed_seconds;
    }
    summary.context_cost         = context_cost.summary;
    return ConstructedTarget{.active            = ActiveTarget(std::move(instance)),
                             .load              = std::move(summary),
                             .sampling_defaults = sampling_defaults,
                             .context_cost      = std::move(context_cost.model)};
}

} // namespace

LoadedQwen3_6_27B::LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model,
                                     const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_27B::make_frontend(*model, options)) {}

LoadedQwen3_6_27B::~LoadedQwen3_6_27B() = default;

Qwen3_6_27BInstance::Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                                         runtime::KvCapacityResolution resolution,
                                         Qwen3_6_27B::SequencePlan sequence_plan,
                                         DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_27B::create_program(*loaded->model, std::move(sequence_plan), device)) {}

Qwen3_6_27BInstance::~Qwen3_6_27BInstance() = default;

LoadedQwen3_6_35BA3B::LoadedQwen3_6_35BA3B(
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model, const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_35BA3B::make_frontend(*model, options)) {}

LoadedQwen3_6_35BA3B::~LoadedQwen3_6_35BA3B() = default;

Qwen3_6_35BA3BInstance::Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                                               runtime::KvCapacityResolution resolution,
                                               Qwen3_6_35BA3B::SequencePlan sequence_plan,
                                               DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_35BA3B::create_program(*loaded->model, std::move(sequence_plan), device)) {}

Qwen3_6_35BA3BInstance::~Qwen3_6_35BA3BInstance() = default;

ConstructedTarget construct_target(const EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto load_start = Clock::now();

    artifact::Reader reader(options.artifact_path);
    std::unique_ptr<artifact::Reader> secondary_reader;
    if (options.artifact_read_mode != ArtifactReadMode::Single) {
        secondary_reader = std::make_unique<artifact::Reader>(options.secondary_artifact_path);
        validate_equivalent_artifacts(reader, *secondary_reader);
    }
    const auto& identity = reader.identity();
    if (identity.model_id == Qwen3_6_27B::model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, device, reader, secondary_reader.get(), load_start, Qwen3_6_27B::target_key);
    }
    if (identity.model_id == Qwen3_6_27B::qwen3_8_model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, device, reader, secondary_reader.get(), load_start,
            Qwen3_6_27B::qwen3_8_target_key);
    }
    if (identity.model_id == Qwen3_6_35BA3B::model_id) {
        return construct_registered<Qwen3_6_35BA3B, LoadedQwen3_6_35BA3B, Qwen3_6_35BA3BInstance>(
            options, device, reader, secondary_reader.get(), load_start,
            Qwen3_6_35BA3B::target_key);
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' has no registered target for this device");
}

} // namespace ninfer::targets
