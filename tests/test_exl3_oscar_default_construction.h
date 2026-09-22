#pragma once

// Bounded constructor-only oracle for dependency-guarded OSCAR defaults.
// It intentionally does not load model weights or create target contexts.

struct OscarDefaultEnvironmentScope {
    struct Saved {
        std::string name;
        std::string value;
        bool present = false;
    };
    std::vector<Saved> saved;

    ~OscarDefaultEnvironmentScope() {
        for (auto it = saved.rbegin(); it != saved.rend(); ++it) {
            _putenv_s(it->name.c_str(), it->present ? it->value.c_str() : "");
        }
    }

    void remember(const char* name) {
        for (const auto& item : saved) if (item.name == name) return;
        const char* value = std::getenv(name);
        saved.push_back({name, value != nullptr ? std::string(value) : std::string{}, value != nullptr});
    }

    void set(const char* name, const char* value) {
        remember(name);
        require(_putenv_s(name, value != nullptr ? value : "") == 0,
                std::string("OSCAR constructor environment set failed: ") + name);
    }

    void unset(const char* name) { set(name, nullptr); }
};

struct OscarDefaultConstructionRecord {
    std::string name;
    int width = 0;
    bool expected_success = false;
    bool observed_success = false;
    std::size_t free_before = 0;
    std::size_t free_after_create = 0;
    std::size_t free_after_destroy = 0;
    std::size_t total_bytes = 0;
    std::uint64_t workspace_bytes = 0;
    std::uint64_t resident_cache_bytes = 0;
    std::string error;
};

inline void oscar_default_set_base(OscarDefaultEnvironmentScope& scope, int width) {
    require(width == 128 || width == 1024, "OSCAR constructor width must be 128 or 1024");
    scope.set("NINFER_EXL3_PREFILL_BATCH_ROTATIONS", "1");
    // The wide ladder is cumulative: width 1024 must still enable 128.
    scope.set("NINFER_EXL3_PREFILL_WIDE128", "1");
    scope.set("NINFER_EXL3_PREFILL_WIDE256", width == 1024 ? "1" : "0");
    scope.set("NINFER_EXL3_PREFILL_WIDE512", width == 1024 ? "1" : "0");
    scope.set("NINFER_EXL3_PREFILL_WIDE1024", width == 1024 ? "1" : "0");
    for (const char* name : {
             "NINFER_OSCAR_PREFILL_COALESCED_ROTATIONS",
             "NINFER_OSCAR_PREFILL_QUERY_PARALLEL",
             "NINFER_OSCAR_PREFILL_PACKED_BYTE4",
             "NINFER_OSCAR_PREFILL_PARALLEL_MERGE",
             "NINFER_OSCAR_PREFILL_PARALLEL_SCORES"}) {
        scope.unset(name);
    }
}

inline void run_oscar_default_construction(const std::string& output_path) {
    require(!output_path.empty(), "oscardefaultconstruction needs OUT");
    require(!std::filesystem::exists(output_path),
            "oscardefaultconstruction output must be new");
    require(std::filesystem::create_directories(output_path),
            "oscardefaultconstruction cannot create OUT directory");
    const auto cases_path = std::filesystem::path(output_path) / "cases.csv";
    const auto result_path = std::filesystem::path(output_path) / "result.txt";
    const std::string asset_dir = env("NINFER_OSCAR_ROTATION_ASSET_DIR");
    const std::string compat_manifest = env("NINFER_OSCAR_COMPAT_MANIFEST");
    require(!asset_dir.empty() && !compat_manifest.empty(),
            "oscardefaultconstruction needs canonical OSCAR asset and compat manifest");

    const auto rotations = ninfer::exl3::exl3_oscar_load_rotations(asset_dir, compat_manifest);
    std::vector<OscarDefaultConstructionRecord> records;
    std::ofstream out(cases_path);
    require(out.good(), "oscardefaultconstruction cannot open OUT");
    out << "case,width,expected_success,observed_success,free_before,free_after_create,free_after_destroy,total_bytes,workspace_bytes,resident_cache_bytes,free_delta,error\n";
    auto write_record = [&](const OscarDefaultConstructionRecord& record) {
        const auto free_delta = static_cast<std::int64_t>(record.free_after_create) -
                                static_cast<std::int64_t>(record.free_before);
        out << record.name << ',' << record.width << ',' << (record.expected_success ? 1 : 0)
            << ',' << (record.observed_success ? 1 : 0) << ',' << record.free_before << ','
            << record.free_after_create << ',' << record.free_after_destroy << ','
            << record.total_bytes << ',' << record.workspace_bytes << ','
            << record.resident_cache_bytes << ',' << free_delta << ',' << record.error << '\n';
        out.flush();
        require(out.good(), "oscardefaultconstruction case output failed");
    };
    auto run_case = [&](const std::string& name, int width, bool expected_success, auto configure) {
        OscarDefaultEnvironmentScope scope;
        oscar_default_set_base(scope, width);
        configure(scope);
        OscarDefaultConstructionRecord record;
        record.name = name;
        record.width = width;
        record.expected_success = expected_success;

        cuda_check(cudaDeviceSynchronize(), "OSCAR constructor prewarm sync");
        cuda_check(cudaMemGetInfo(&record.free_before, &record.total_bytes),
                   "OSCAR constructor free-before query");
        ninfer::exl3::Exl3OscarTelemetry telemetry{};
        std::unique_ptr<ninfer::exl3::Exl3OscarContext> context;
        try {
            context = ninfer::exl3::Exl3OscarContext::create(rotations, width, &telemetry);
        } catch (const std::invalid_argument& error) {
            record.error = error.what();
            // Rejected constructor paths fail before a context exists; keep
            // the create snapshot neutral while the destroy query below
            // still proves that the free baseline was restored.
            record.free_after_create = record.free_before;
        }
        record.observed_success = context != nullptr;
        if (record.observed_success) {
            record.workspace_bytes = telemetry.workspace_bytes;
            record.resident_cache_bytes = telemetry.resident_cache_bytes;
            cuda_check(cudaDeviceSynchronize(), "OSCAR constructor create sync");
            cuda_check(cudaMemGetInfo(&record.free_after_create, &record.total_bytes),
                       "OSCAR constructor free-after-create query");
            context.reset();
        }
        cuda_check(cudaDeviceSynchronize(), "OSCAR constructor destroy sync");
        std::size_t free_after_destroy = 0;
        cuda_check(cudaMemGetInfo(&free_after_destroy, &record.total_bytes),
                   "OSCAR constructor free-after-destroy query");
        record.free_after_destroy = free_after_destroy;
        records.push_back(record);
        write_record(records.back());
        const char* expected_error = nullptr;
        if (!expected_success) {
            expected_error = name.rfind("query_one_", 0) == 0
                ? "OSCAR query-parallel prefill requires batch rotations and parallel scores/merge"
                : name.rfind("packed_one_", 0) == 0
                    ? "OSCAR packed-byte4 prefill requires query-parallel" : nullptr;
            require(expected_error != nullptr, name + " has no expected rejection contract");
            require(!record.observed_success && record.error == expected_error,
                    name + " unexpected rejection: " + record.error);
        } else {
            require(record.observed_success, name + " unexpectedly rejected: " + record.error);
        }
        require(record.free_after_destroy == record.free_before,
                name + " constructor destruction did not restore free baseline");
    };

    const auto configure_unset = [](OscarDefaultEnvironmentScope&) {};
    const auto configure_all1 = [](OscarDefaultEnvironmentScope& scope) {
        scope.set("NINFER_OSCAR_PREFILL_COALESCED_ROTATIONS", "1");
        scope.set("NINFER_OSCAR_PREFILL_QUERY_PARALLEL", "1");
        scope.set("NINFER_OSCAR_PREFILL_PACKED_BYTE4", "1");
        scope.set("NINFER_OSCAR_PREFILL_PARALLEL_MERGE", "1");
        scope.set("NINFER_OSCAR_PREFILL_PARALLEL_SCORES", "1");
    };
    {
        // Complete one constructor lifetime before recording matrix memory.
        OscarDefaultEnvironmentScope scope;
        oscar_default_set_base(scope, 1024);
        configure_all1(scope);
        cuda_check(cudaFree(0), "OSCAR constructor prewarm allocation");
        cuda_check(cudaDeviceSynchronize(), "OSCAR constructor prewarm allocation sync");
        ninfer::exl3::Exl3OscarTelemetry telemetry{};
        auto context = ninfer::exl3::Exl3OscarContext::create(rotations, 1024, &telemetry);
        cuda_check(cudaDeviceSynchronize(), "OSCAR constructor prewarm create sync");
        context.reset();
        cuda_check(cudaDeviceSynchronize(), "OSCAR constructor prewarm destroy sync");
    }
    for (const int width : {128, 1024}) {
        run_case("all_unset", width, true, configure_unset);
        run_case("all_one", width, true, configure_all1);
        run_case("coalesced_zero", width, true, [](OscarDefaultEnvironmentScope& scope) {
            scope.set("NINFER_OSCAR_PREFILL_COALESCED_ROTATIONS", "0");
        });
        run_case("query_zero", width, true, [](OscarDefaultEnvironmentScope& scope) {
            scope.set("NINFER_OSCAR_PREFILL_QUERY_PARALLEL", "0");
        });
        run_case("packed_zero", width, true, [](OscarDefaultEnvironmentScope& scope) {
            scope.set("NINFER_OSCAR_PREFILL_PACKED_BYTE4", "0");
        });
        run_case("batch_zero_dependents_unset", width, true, [](OscarDefaultEnvironmentScope& scope) {
            scope.set("NINFER_EXL3_PREFILL_BATCH_ROTATIONS", "0");
        });
        run_case("merge_zero_query_unset", width, true, [](OscarDefaultEnvironmentScope& scope) {
            scope.set("NINFER_OSCAR_PREFILL_PARALLEL_MERGE", "0");
        });
        run_case("scores_zero_query_unset", width, true, [](OscarDefaultEnvironmentScope& scope) {
            scope.set("NINFER_OSCAR_PREFILL_PARALLEL_SCORES", "0");
        });
        run_case("merge_other_query_unset", width, true, [](OscarDefaultEnvironmentScope& scope) {
            scope.set("NINFER_OSCAR_PREFILL_PARALLEL_MERGE", "other");
        });
        run_case("scores_other_query_unset", width, true, [](OscarDefaultEnvironmentScope& scope) {
            scope.set("NINFER_OSCAR_PREFILL_PARALLEL_SCORES", "other");
        });
        run_case("coalesced_other", width, true, [](OscarDefaultEnvironmentScope& scope) {
            scope.set("NINFER_OSCAR_PREFILL_COALESCED_ROTATIONS", "other");
        });
        run_case("query_other", width, true, [](OscarDefaultEnvironmentScope& scope) {
            scope.set("NINFER_OSCAR_PREFILL_QUERY_PARALLEL", "other");
        });
        run_case("packed_other", width, true, [](OscarDefaultEnvironmentScope& scope) {
            scope.set("NINFER_OSCAR_PREFILL_PACKED_BYTE4", "other");
        });
        run_case("query_one_merge_zero_reject", width, false, [](OscarDefaultEnvironmentScope& scope) {
            scope.set("NINFER_OSCAR_PREFILL_QUERY_PARALLEL", "1");
            scope.set("NINFER_OSCAR_PREFILL_PARALLEL_MERGE", "0");
        });
        run_case("query_one_scores_zero_reject", width, false, [](OscarDefaultEnvironmentScope& scope) {
            scope.set("NINFER_OSCAR_PREFILL_QUERY_PARALLEL", "1");
            scope.set("NINFER_OSCAR_PREFILL_PARALLEL_SCORES", "0");
        });
        run_case("query_one_batch_zero_reject", width, false, [](OscarDefaultEnvironmentScope& scope) {
            scope.set("NINFER_EXL3_PREFILL_BATCH_ROTATIONS", "0");
            scope.set("NINFER_OSCAR_PREFILL_QUERY_PARALLEL", "1");
        });
        run_case("packed_one_query_zero_reject", width, false, [](OscarDefaultEnvironmentScope& scope) {
            scope.set("NINFER_OSCAR_PREFILL_PACKED_BYTE4", "1");
            scope.set("NINFER_OSCAR_PREFILL_QUERY_PARALLEL", "0");
        });
        run_case("packed_one_batch_zero_reject", width, false, [](OscarDefaultEnvironmentScope& scope) {
            scope.set("NINFER_EXL3_PREFILL_BATCH_ROTATIONS", "0");
            scope.set("NINFER_OSCAR_PREFILL_PACKED_BYTE4", "1");
        });
    }

    auto find = [&](const std::string& name, int width) -> const OscarDefaultConstructionRecord& {
        for (const auto& record : records)
            if (record.name == name && record.width == width) return record;
        throw std::runtime_error("OSCAR constructor record missing: " + name);
    };
    constexpr std::uint64_t kCoalescedBytes = 4194304ULL;
    constexpr std::uint64_t kQueryBytes = 26411008ULL;
    const auto allocation_delta = [](const OscarDefaultConstructionRecord& record) {
        require(record.observed_success && record.free_before >= record.free_after_create,
                record.name + " missing valid allocation memory sample");
        return record.free_before - record.free_after_create;
    };
    for (const int width : {128, 1024}) {
        const auto& all_unset = find("all_unset", width);
        const auto& all_one = find("all_one", width);
        const auto all_unset_alloc = allocation_delta(all_unset);
        require(all_one.workspace_bytes == all_unset.workspace_bytes &&
                    all_one.resident_cache_bytes == all_unset.resident_cache_bytes,
                "all-unset/all-one OSCAR logical bytes differ");
        require(all_unset_alloc == allocation_delta(all_one),
                "all-unset/all-one OSCAR free allocation deltas differ");
        const auto& coalesced_zero = find("coalesced_zero", width);
        const auto& query_zero = find("query_zero", width);
        const auto& packed_zero = find("packed_zero", width);
        require(find("coalesced_zero", width).workspace_bytes + kCoalescedBytes ==
                    all_unset.workspace_bytes,
                "coalesced dependency workspace delta mismatch");
        require(find("query_zero", width).workspace_bytes + kQueryBytes ==
                    all_unset.workspace_bytes,
                "query dependency workspace delta mismatch");
        require(find("packed_zero", width).workspace_bytes == all_unset.workspace_bytes,
                "packed default unexpectedly changes logical workspace");
        require(find("merge_zero_query_unset", width).workspace_bytes + kQueryBytes ==
                    all_unset.workspace_bytes,
                "merge dependency workspace delta mismatch");
        require(find("scores_zero_query_unset", width).workspace_bytes + kQueryBytes ==
                    all_unset.workspace_bytes,
                "scores dependency workspace delta mismatch");
        // `free_before - free_after_create` is a physical CUDA allocator
        // observation. It is recorded in cases.csv and the all-unset/all-one
        // equality above is required, but it is not required to equal the
        // logical telemetry delta: cudaMemGetInfo includes allocator/page
        // granularity and unrelated driver bookkeeping. The exact
        // coalesced/query byte contracts are asserted through workspace_bytes
        // above; retain the physical samples for audit without equating the
        // two accounting domains.
        const auto& coalesced_other = find("coalesced_other", width);
        const auto& query_other = find("query_other", width);
        const auto& packed_other = find("packed_other", width);
        const auto& merge_zero = find("merge_zero_query_unset", width);
        const auto& scores_zero = find("scores_zero_query_unset", width);
        const auto& merge_other = find("merge_other_query_unset", width);
        const auto& scores_other = find("scores_other_query_unset", width);
        require(coalesced_other.workspace_bytes == coalesced_zero.workspace_bytes &&
                    coalesced_other.resident_cache_bytes == coalesced_zero.resident_cache_bytes &&
                    allocation_delta(coalesced_other) == allocation_delta(coalesced_zero),
                "coalesced other-value path differs from zero");
        require(query_other.workspace_bytes == query_zero.workspace_bytes &&
                    query_other.resident_cache_bytes == query_zero.resident_cache_bytes &&
                    allocation_delta(query_other) == allocation_delta(query_zero),
                "query other-value path differs from zero");
        require(packed_other.workspace_bytes == packed_zero.workspace_bytes &&
                    packed_other.resident_cache_bytes == packed_zero.resident_cache_bytes &&
                    allocation_delta(packed_other) == allocation_delta(packed_zero),
                "packed other-value path differs from zero");
        require(merge_other.workspace_bytes == merge_zero.workspace_bytes &&
                    merge_other.resident_cache_bytes == merge_zero.resident_cache_bytes &&
                    allocation_delta(merge_other) == allocation_delta(merge_zero),
                "merge other-value path differs from zero");
        require(scores_other.workspace_bytes == scores_zero.workspace_bytes &&
                    scores_other.resident_cache_bytes == scores_zero.resident_cache_bytes &&
                    allocation_delta(scores_other) == allocation_delta(scores_zero),
                "scores other-value path differs from zero");
    }

    out.flush();
    require(out.good(), "oscardefaultconstruction output failed");
    std::ofstream result(result_path);
    require(result.good(), "oscardefaultconstruction cannot open result");
    result << "OSCAR_DEFAULT_CONSTRUCTION PASS cases=" << records.size()
           << " widths=128,1024 coalesced_delta=4194304 query_delta=26411008"
           << " destruction_baseline=1\n";
    result.flush();
    require(result.good(), "oscardefaultconstruction result failed");
    std::cout << "OSCAR_DEFAULT_CONSTRUCTION PASS cases=" << records.size()
              << " widths=128,1024 coalesced_delta=4194304 query_delta=26411008"
              << " destruction_baseline=1\n";
}
