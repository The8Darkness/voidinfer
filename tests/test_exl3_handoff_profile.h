#pragma once

// Qualification-only handoff profiler. The owner exists only when the profile
// prefix is set, so ordinary acceptance creates no events, files, or queries.
class HandoffProfile {
public:
    struct Handle { int slot = -1; };

    explicit HandoffProfile(const std::filesystem::path& prefix) {
        require(!prefix.empty(), "handoff profile prefix is empty");
        stages_path_ = prefix.string() + "-stages.csv";
        memory_path_ = prefix.string() + "-memory.csv";
        metadata_path_ = prefix.string() + "-metadata.csv";
        for (const auto& path : {stages_path_, memory_path_, metadata_path_}) {
            require(!std::filesystem::exists(path),
                    "handoff profile refuses existing output: " + path.string());
            if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
        }
        stages_.open(stages_path_); memory_.open(memory_path_); metadata_.open(metadata_path_);
        require(stages_.good() && memory_.good() && metadata_.good(),
                "cannot open handoff profile outputs");
        stages_.imbue(std::locale::classic()); memory_.imbue(std::locale::classic());
        metadata_.imbue(std::locale::classic());
        stages_ << "round,scope,stage,cuda_ms,wall_ms\n";
        memory_ << "sample,round,cuda_free_bytes,cuda_total_bytes,cuda_used_since_ready_bytes,"
                   "target_model_bytes,draft_weight_bytes,draft_kv_bytes,draft_scratch_bytes,"
                   "draft_linear_workspace_omitted_bytes,draft_ring_bytes,tap_stage_bytes,"
                   "context_persistent_bytes,oscar_resident_bytes,oscar_workspace_bytes,"
                   "oscar_omitted_bytes,transaction_bytes,continuation_bytes,known_device_bytes,"
                   "unattributed_device_bytes,working_set_bytes,peak_working_set_bytes,"
                   "private_commit_bytes,pagefile_commit_bytes,peak_pagefile_commit_bytes,"
                   "page_faults,physical_total_bytes,physical_available_bytes,"
                   "commit_limit_bytes,commit_available_bytes\n";
        metadata_ << "key,value\n";
        int runtime = 0, driver = 0, device = 0;
        cuda_check(cudaRuntimeGetVersion(&runtime), "handoff profile runtime version");
        cuda_check(cudaDriverGetVersion(&driver), "handoff profile driver version");
        cuda_check(cudaGetDevice(&device), "handoff profile current device");
        cudaDeviceProp prop{};
        cuda_check(cudaGetDeviceProperties(&prop, device), "handoff profile device properties");
        MEMORYSTATUSEX global{}; global.dwLength = sizeof(global);
        require(GlobalMemoryStatusEx(&global) != 0,
                "handoff profile GlobalMemoryStatusEx failed");
        metadata_ << "cuda_runtime_version," << runtime << "\n"
                  << "cuda_driver_version," << driver << "\n"
                  << "cuda_device," << device << "\n"
                  << "cuda_device_name," << prop.name << "\n"
                  << "cuda_sm_count," << prop.multiProcessorCount << "\n"
                  << "cuda_compute_major," << prop.major << "\n"
                  << "cuda_compute_minor," << prop.minor << "\n"
                  << "system_physical_total_bytes," << global.ullTotalPhys << "\n"
                  << "system_physical_available_bytes," << global.ullAvailPhys << "\n"
                  << "system_commit_limit_bytes," << global.ullTotalPageFile << "\n"
                  << "system_commit_available_bytes," << global.ullAvailPageFile << "\n"
                  << "draft_linear_workspace_omitted_bytes,48349184\n"
                  << "oscar_allocation_omitted_bytes,9486336\n"
                  << "measurement_scope,instrumented_diagnostic_not_release_performance\n"
                  << "event_resolution,single_stream_sync_outside_round_stopwatch\n"
                  << "round_memory_samples,first_round_only\n"
                  << "prefill_internal_breakdown,NOT_INSTRUMENTED\n"
                  << "control_decode_internal_breakdown,NOT_INSTRUMENTED\n"
                  << "target_profile_decode,NOT_INSTRUMENTED\n";
        metadata_.flush();
        stack_.reserve(slots_.size());
        try {
            for (auto& slot : slots_) {
                cuda_check(cudaEventCreate(&slot.begin),
                           "handoff profile create begin event");
                cuda_check(cudaEventCreate(&slot.end),
                           "handoff profile create end event");
            }
        } catch (...) {
            for (auto& slot : slots_) {
                if (slot.begin) { cudaEventDestroy(slot.begin); slot.begin = nullptr; }
                if (slot.end) { cudaEventDestroy(slot.end); slot.end = nullptr; }
            }
            throw;
        }
    }
    ~HandoffProfile() noexcept {
        while (!stack_.empty()) {
            nvtxRangePop();
            stack_.pop_back();
        }
        for (auto& slot : slots_) {
            if (slot.begin) cudaEventDestroy(slot.begin);
            if (slot.end) cudaEventDestroy(slot.end);
        }
    }
    HandoffProfile(const HandoffProfile&) = delete;
    HandoffProfile& operator=(const HandoffProfile&) = delete;

    Handle begin(int round, const char* scope, const char* stage, cudaStream_t stream = nullptr) {
        require(used_ < static_cast<int>(slots_.size()), "handoff profile event pool exhausted");
        auto& slot = slots_[static_cast<std::size_t>(used_)];
        slot.round = round; slot.scope = scope; slot.stage = stage;
        slot.nvtx_label = "handoff." + slot.scope + "." + slot.stage;
        slot.wall_begin = std::chrono::steady_clock::now(); slot.ended = false;
        cuda_check(cudaEventRecord(slot.begin, stream), "handoff profile record begin");
        nvtxRangePushA(slot.nvtx_label.c_str());
        stack_.push_back(used_);
        return Handle{used_++};
    }
    void end(Handle handle, cudaStream_t stream = nullptr) {
        require(handle.slot >= 0 && handle.slot < used_ && !stack_.empty() &&
                    stack_.back() == handle.slot, "handoff profile stage nesting mismatch");
        auto& slot = slots_[static_cast<std::size_t>(handle.slot)];
        slot.wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - slot.wall_begin).count();
        cuda_check(cudaEventRecord(slot.end, stream), "handoff profile record end");
        slot.ended = true; stack_.pop_back(); nvtxRangePop();
    }
    // One diagnostic-only synchronization resolves the complete event batch;
    // there is no per-stage synchronization.
    void resolve(cudaStream_t stream = nullptr) {
        require(stack_.empty(), "handoff profile has an open stage");
        cuda_check(cudaStreamSynchronize(stream), "handoff profile resolve events");
        for (int i = 0; i < used_; ++i) {
            auto& slot = slots_[static_cast<std::size_t>(i)];
            require(slot.ended, "handoff profile contains an unfinished stage");
            float cuda_ms = 0.0F;
            cuda_check(cudaEventElapsedTime(&cuda_ms, slot.begin, slot.end),
                       "handoff profile elapsed time");
            stages_ << slot.round << ',' << slot.scope << ',' << slot.stage << ','
                    << std::fixed << std::setprecision(6) << cuda_ms << ',' << slot.wall_ms << "\n";
        }
        stages_.flush(); used_ = 0;
    }

    void sample_memory(const char* sample, int round, const Exl3TextModel* target,
                       const Exl3Dflash2DraftModel* draft, const Exl3TextContext* context,
                       std::size_t tap_stage_bytes = 0, cudaStream_t stream = nullptr) {
        cuda_check(cudaStreamSynchronize(stream), "handoff profile memory boundary");
        std::size_t free_bytes = 0, total_bytes = 0;
        cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "handoff profile memory info");
        if (!baseline_set_) { baseline_free_ = free_bytes; baseline_set_ = true; }
        const long long used = static_cast<long long>(baseline_free_) -
            static_cast<long long>(free_bytes);
        const std::size_t target_bytes = target ? target->model_bytes() : 0;
        const std::size_t draft_weight = draft ? draft->weight_bytes() : 0;
        const std::size_t draft_kv = draft ? draft->kv_bytes() : 0;
        const std::size_t draft_scratch = draft ? draft->scratch_bytes() : 0;
        const std::size_t draft_ring = draft ? draft->ring_bytes() : 0;
        const std::size_t context_persistent = context ? context->persistent_bytes() : 0;
        const std::size_t transaction = context ? context->transaction_bytes() : 0;
        const std::size_t continuation = context ? context->continuation_bytes() : 0;
        std::size_t oscar_resident = 0, oscar_workspace = 0, oscar_omitted = 0;
        if (context && context->oscar_enabled()) {
            const auto& telemetry = context->oscar_telemetry();
            oscar_resident = telemetry.resident_cache_bytes;
            oscar_workspace = telemetry.workspace_bytes;
            oscar_omitted = 9486336;
        }
        const std::size_t draft_workspace_omitted = draft ? 48349184 : 0;
        // transaction/continuation are component columns of context_persistent,
        // not additional allocations; do not double count them in known.
        const std::size_t known = target_bytes + draft_weight + draft_kv + draft_scratch +
            draft_workspace_omitted + draft_ring + tap_stage_bytes + context_persistent +
            oscar_resident + oscar_workspace + oscar_omitted;
        const long long unattributed = used - static_cast<long long>(known);
        PROCESS_MEMORY_COUNTERS_EX counters{}; counters.cb = sizeof(counters);
        require(K32GetProcessMemoryInfo(GetCurrentProcess(),
                    reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters)) != 0,
                "handoff profile K32GetProcessMemoryInfo failed");
        MEMORYSTATUSEX global{}; global.dwLength = sizeof(global);
        require(GlobalMemoryStatusEx(&global) != 0,
                "handoff profile GlobalMemoryStatusEx failed");
        memory_ << sample << ',' << round << ',' << free_bytes << ',' << total_bytes << ','
                << used << ',' << target_bytes << ',' << draft_weight << ',' << draft_kv << ','
                << draft_scratch << ',' << draft_workspace_omitted << ',' << draft_ring << ','
                << tap_stage_bytes << ',' << context_persistent << ',' << oscar_resident << ','
                << oscar_workspace << ',' << oscar_omitted << ',' << transaction << ','
                << continuation << ',' << known << ',' << unattributed << ','
                << counters.WorkingSetSize << ',' << counters.PeakWorkingSetSize << ','
                << counters.PrivateUsage << ',' << counters.PagefileUsage << ','
                << counters.PeakPagefileUsage << ',' << counters.PageFaultCount << ','
                << global.ullTotalPhys << ',' << global.ullAvailPhys << ','
                << global.ullTotalPageFile << ',' << global.ullAvailPageFile << "\n";
        memory_.flush();
    }

private:
    struct Slot {
        cudaEvent_t begin = nullptr, end = nullptr;
        int round = -1;
        std::string scope, stage, nvtx_label;
        std::chrono::steady_clock::time_point wall_begin{};
        double wall_ms = 0.0;
        bool ended = false;
    };
    std::array<Slot, 32> slots_{};
    std::vector<int> stack_;
    int used_ = 0;
    bool baseline_set_ = false;
    std::size_t baseline_free_ = 0;
    std::filesystem::path stages_path_, memory_path_, metadata_path_;
    std::ofstream stages_, memory_, metadata_;
};
