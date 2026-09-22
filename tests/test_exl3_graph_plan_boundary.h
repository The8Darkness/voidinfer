#pragma once

struct GraphPlanBoundaryOracle {
    std::array<std::uint32_t, 9> plan{};
    std::array<std::int32_t, 5> counts{};
};

std::int64_t graph_plan_boundary_begin(std::int64_t context) {
    return context <= 64 ? context : std::max<std::int64_t>(64, context - 256);
}

GraphPlanBoundaryOracle graph_plan_boundary_oracle(int old_position) {
    const std::int64_t old_context = old_position;
    const std::int64_t new_context = old_context + 1;
    const std::int64_t old_begin = graph_plan_boundary_begin(old_context);
    const std::int64_t final_begin = graph_plan_boundary_begin(new_context);
    const std::int64_t old_aging_end = std::min(final_begin, old_context);
    const std::int64_t old_aging = std::max<std::int64_t>(0, old_aging_end - old_begin);
    const std::int64_t new_hist_begin = std::max<std::int64_t>(64, old_context);
    const std::int64_t new_hist_end = std::min(new_context, final_begin);
    const std::int64_t new_hist = std::max<std::int64_t>(0, new_hist_end - new_hist_begin);
    GraphPlanBoundaryOracle result;
    result.plan = {
        1U, static_cast<std::uint32_t>(old_context), static_cast<std::uint32_t>(old_context),
        static_cast<std::uint32_t>(old_begin), static_cast<std::uint32_t>(old_begin & 255),
        static_cast<std::uint32_t>(final_begin), static_cast<std::uint32_t>(new_hist),
        static_cast<std::uint32_t>(old_aging), static_cast<std::uint32_t>(final_begin & 255)};
    const std::int64_t prefix = std::min<std::int64_t>(new_context, 64);
    result.counts = {
        static_cast<std::int32_t>(prefix),
        static_cast<std::int32_t>(final_begin - prefix),
        static_cast<std::int32_t>(new_context - final_begin),
        static_cast<std::int32_t>(final_begin & 255),
        static_cast<std::int32_t>(new_context - 1)};
    return result;
}

std::string graph_plan_boundary_words(const std::array<std::uint32_t, 9>& values) {
    std::ostringstream out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    return out.str();
}

std::string graph_plan_boundary_signed(const std::array<std::int32_t, 5>& values) {
    std::ostringstream out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    return out.str();
}

void run_graph_plan_boundary(const std::string& out_path) {
    require(!out_path.empty(), "graphplanboundary requires NINFER_E5A4_OUT");
    require(!std::filesystem::exists(out_path),
            "graphplanboundary output must be a new file");
    std::ofstream report(out_path);
    require(report.good(), "graphplanboundary cannot open output");
    report << "mode=graphplanboundary\n"
           << "required_environment=NINFER_E5A4_MODE=graphplanboundary,NINFER_E5A4_OUT=new; no target or draft path\n"
           << "kernel_scope=actual oscar_exl3_plan_kernel only; no cache append, cache write, decode, model load, or graph lifecycle\n"
           << "oracle_scope=independent canonical begin=max(64,context-256) after64; raw unsigned plan words and signed count words retained\n"
           << "positions=0..1023\n";
    bool status_written = false;
    try {
        cuda_check(cudaSetDevice(0), "graphplanboundary set device");
        cuda_check(cudaDeviceSynchronize(), "graphplanboundary initial sync");
        int mismatches = 0;
        for (int old_position = 0; old_position <= 1023; ++old_position) {
            const auto actual = ninfer::exl3::oscar_plan_boundary_observe_for_test(old_position);
            const auto expected = graph_plan_boundary_oracle(old_position);
            if (actual.plan != expected.plan || actual.counts != expected.counts) {
                ++mismatches;
                report << "mismatch position=" << old_position
                       << " actual_plan=" << graph_plan_boundary_words(actual.plan)
                       << " expected_plan=" << graph_plan_boundary_words(expected.plan)
                       << " actual_counts=" << graph_plan_boundary_signed(actual.counts)
                       << " expected_counts=" << graph_plan_boundary_signed(expected.counts)
                       << '\n';
            }
        }
        report << "mismatch_count=" << mismatches << '\n';
        if (mismatches != 0) {
            report << "reportStatus=FAIL\n"
                   << "failure=actual plan kernel disagrees with independent canonical oracle\n";
            report.flush();
            status_written = true;
            std::cout << "GRAPH_PLAN_BOUNDARY FAIL mismatches=" << mismatches << "\n";
            throw std::runtime_error("graphplanboundary oracle mismatch");
        }
        report << "reportStatus=PASS\n";
        report.flush();
        status_written = true;
        std::cout << "GRAPH_PLAN_BOUNDARY PASS\n";
    } catch (const std::exception& error) {
        if (!status_written) {
            report << "reportStatus=FAIL\n"
                   << "failure=" << error.what() << '\n';
            report.flush();
        }
        throw;
    }
}
