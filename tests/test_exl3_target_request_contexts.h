#pragma once

#include <cstddef>

// Bounded eager target-only request-count measurement.  This helper is kept
// separate from the legacy one-context targetrequest path so its CSV contract
// and serial round-robin scheduler cannot alter the existing C1 bridge.

struct TargetRequestContextRow {
    int context_id = -1;
    std::vector<std::int64_t> tokens;
    double context_create_ms = 0.0;
    double oscar_attach_ms = 0.0;
    double ingestion_wall_ms = 0.0;
    double seed_wall_ms = 0.0;
    double first_output_ms = 0.0;
    double last_output_ms = 0.0;
};

struct TargetRequestRoundRow {
    int round = -1;
    double wall_ms = 0.0;
};

void run_targetrequest_contexts(
    Exl3TextModel& target, const std::vector<std::int64_t>& source_ids,
    int prefix_count, int outputs, int context_count, std::size_t source_offset,
    const std::string& summary_path, const std::string& tokens_path,
    const std::string& rounds_path) {
    require(context_count == 1 || context_count == 2 || context_count == 4 || context_count == 8,
            "targetrequest context count must be 1, 2, 4, or 8");
    require(prefix_count >= 32 && outputs >= 1 && target.max_context() == 1024,
            "targetrequest context benchmark extent");
    require(source_offset <= source_ids.size(), "targetrequest source offset extent");
    const std::size_t stride = 512;
    require(static_cast<std::size_t>(context_count - 1) <=
                (source_ids.size() - source_offset) / stride,
            "targetrequest source slices do not fit");
    require(static_cast<std::size_t>(prefix_count) <=
                source_ids.size() - source_offset - static_cast<std::size_t>(context_count - 1) * stride,
            "targetrequest source slice prefix extent");
    require(static_cast<std::int64_t>(prefix_count) + outputs - 1 <= target.max_context(),
            "targetrequest context benchmark max context extent");

    std::vector<std::vector<std::int64_t>> prefixes;
    prefixes.reserve(static_cast<std::size_t>(context_count));
    for (int stream = 0; stream < context_count; ++stream) {
        const std::size_t begin = source_offset + static_cast<std::size_t>(stream) * stride;
        std::vector<std::int64_t> prefix(
            source_ids.begin() + static_cast<std::ptrdiff_t>(begin),
            source_ids.begin() + static_cast<std::ptrdiff_t>(begin + prefix_count));
        require(prefix.size() == static_cast<std::size_t>(prefix_count),
                "targetrequest source slice size");
        for (const auto id : prefix)
            require(id >= 0 && id < kVocab, "targetrequest source slice token extent");
        require(prefix.back() != kMaskToken,
                "targetrequest source slice cannot end in mask");
        prefixes.push_back(std::move(prefix));
    }

    using Clock = std::chrono::steady_clock;
    const auto resident_batch_start = Clock::now();
    std::vector<std::unique_ptr<Exl3TextContext>> contexts;
    contexts.reserve(static_cast<std::size_t>(context_count));
    std::vector<TargetRequestContextRow> rows(static_cast<std::size_t>(context_count));
    for (int stream = 0; stream < context_count; ++stream) {
        rows[static_cast<std::size_t>(stream)].context_id = stream;
        const auto create_begin = Clock::now();
        contexts.push_back(target.create_context(false));
        const auto create_end = Clock::now();
        rows[static_cast<std::size_t>(stream)].context_create_ms =
            std::chrono::duration<double, std::milli>(create_end - create_begin).count();
        const auto attach_begin = Clock::now();
        require(contexts.back()->try_enable_oscar_from_environment(),
                "targetrequest context benchmark canonical OSCAR attachment");
        const auto attach_end = Clock::now();
        rows[static_cast<std::size_t>(stream)].oscar_attach_ms =
            std::chrono::duration<double, std::milli>(attach_end - attach_begin).count();
    }

    // Serialize each request's prefill and seed before entering the timed
    // round-robin decode stage.  No oracle or profiling work is performed.
    for (int stream = 0; stream < context_count; ++stream) {
        auto& row = rows[static_cast<std::size_t>(stream)];
        auto& context = *contexts[static_cast<std::size_t>(stream)];
        const auto ingest_begin = Clock::now();
        ingest_prefix(context, prefixes[static_cast<std::size_t>(stream)], CommitSink{});
        const auto ingest_end = Clock::now();
        row.ingestion_wall_ms =
            std::chrono::duration<double, std::milli>(ingest_end - ingest_begin).count();
        const auto seed_begin = Clock::now();
        row.tokens.reserve(static_cast<std::size_t>(outputs));
        row.tokens.push_back(sample_target(context));
        const auto seed_end = Clock::now();
        row.seed_wall_ms =
            std::chrono::duration<double, std::milli>(seed_end - seed_begin).count();
        row.first_output_ms =
            std::chrono::duration<double, std::milli>(seed_end - resident_batch_start).count();
        row.last_output_ms = row.first_output_ms;
    }

    const auto decode_stage_begin = Clock::now();
    std::vector<TargetRequestRoundRow> rounds;
    rounds.reserve(static_cast<std::size_t>(std::max(0, outputs - 1)));
    for (int round = 0; round < outputs - 1; ++round) {
        const auto round_begin = Clock::now();
        for (int stream = 0; stream < context_count; ++stream) {
            auto& row = rows[static_cast<std::size_t>(stream)];
            auto& context = *contexts[static_cast<std::size_t>(stream)];
            require(!row.tokens.empty(), "targetrequest missing pending seed");
            context.decode(row.tokens.back());
            row.tokens.push_back(sample_target(context));
            row.last_output_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - resident_batch_start).count();
            require(context.position() == prefix_count + static_cast<int>(row.tokens.size()) - 1,
                    "targetrequest context pending position mismatch");
        }
        rounds.push_back(TargetRequestRoundRow{
            round,
            std::chrono::duration<double, std::milli>(Clock::now() - round_begin).count()});
    }
    const auto complete_batch_end = Clock::now();
    const double decode_stage_wall_ms =
        std::chrono::duration<double, std::milli>(complete_batch_end - decode_stage_begin).count();
    const double complete_batch_last_output_ms =
        std::chrono::duration<double, std::milli>(complete_batch_end - resident_batch_start).count();
    for (const auto& row : rows) {
        require(row.tokens.size() == static_cast<std::size_t>(outputs),
                "targetrequest context emitted token count mismatch");
    }

    std::ofstream summary(summary_path); summary.imbue(std::locale::classic());
    require(summary.good(), "targetrequest contexts cannot open summary");
    summary << "row_kind,context_id,round_index,ingested,emitted,target_decodes,position,pending_token,resident_batch_start_ms,context_create_ms,oscar_attach_ms,ingestion_wall_ms,seed_wall_ms,first_output_ms,last_output_ms,complete_batch_last_output_ms,decode_stage_first_after_all_seeds_ms,decode_stage_last_output_ms,decode_stage_wall_ms,round_wall_ms,decode_tokens_per_second,aggregate_decode_tokens_per_second,scheduler,token_accounting\n";
    summary << std::fixed << std::setprecision(9);
    for (const auto& row : rows) {
        const double stream_decode_ms = row.last_output_ms -
            std::chrono::duration<double, std::milli>(decode_stage_begin - resident_batch_start).count();
        const double stream_tps = stream_decode_ms > 0.0
            ? static_cast<double>(outputs - 1) * 1000.0 / stream_decode_ms : 0.0;
        summary << "stream," << row.context_id << ",-1," << prefix_count << ',' << row.tokens.size() << ','
                << outputs - 1 << ',' << prefix_count + outputs - 1 << ',' << row.tokens.back() << ','
                << 0.0 << ',' << row.context_create_ms << ',' << row.oscar_attach_ms << ','
                << row.ingestion_wall_ms << ',' << row.seed_wall_ms << ',' << row.first_output_ms << ','
                << row.last_output_ms << ',' << complete_batch_last_output_ms << ','
                << std::chrono::duration<double, std::milli>(decode_stage_begin - resident_batch_start).count() << ','
                << row.last_output_ms << ',' << stream_decode_ms << ",0," << stream_tps << ",0,serial_eager_round_robin,emitted_includes_seed;target_decodes=emitted-1\n";
    }
    const double aggregate_tps = decode_stage_wall_ms > 0.0
        ? static_cast<double>(context_count * (outputs - 1)) * 1000.0 / decode_stage_wall_ms : 0.0;
    summary << "aggregate,-1,-1," << context_count * prefix_count << ','
            << context_count * outputs << ',' << context_count * (outputs - 1) << ",-1,-1,"
            << 0.0 << ",0,0,0,0,0,0," << complete_batch_last_output_ms << ','
            << std::chrono::duration<double, std::milli>(decode_stage_begin - resident_batch_start).count() << ','
            << complete_batch_last_output_ms << ',' << decode_stage_wall_ms << ",0,0," << aggregate_tps
            << ",serial_eager_round_robin,emitted_includes_seed;target_decodes=emitted-1\n";
    for (const auto& round : rounds) {
        summary << "round,-1," << round.round << ",0,0,0,-1,-1,0,0,0,0,0,0,0,"
                << complete_batch_last_output_ms << ",0,0,0," << round.wall_ms << ",0,0,serial_eager_round_robin,round_wall_ms\n";
    }
    summary.flush(); require(summary.good(), "targetrequest contexts summary output failed");

    std::ofstream token_file(tokens_path); token_file.imbue(std::locale::classic());
    require(token_file.good(), "targetrequest contexts cannot open token witness");
    token_file << "context_id,ordinal,token_id\n";
    for (const auto& row : rows)
        for (std::size_t ordinal = 0; ordinal < row.tokens.size(); ++ordinal)
            token_file << row.context_id << ',' << ordinal << ',' << row.tokens[ordinal] << '\n';
    token_file.flush(); require(token_file.good(), "targetrequest contexts token output failed");

    std::ofstream round_file(rounds_path); round_file.imbue(std::locale::classic());
    require(round_file.good(), "targetrequest contexts cannot open round output");
    round_file << "round_index,round_wall_ms,contexts,outputs_per_context,scheduler\n";
    round_file << std::fixed << std::setprecision(9);
    for (const auto& round : rounds)
        round_file << round.round << ',' << round.wall_ms << ',' << context_count << ',' << outputs
                   << ",serial_eager_round_robin\n";
    round_file.flush(); require(round_file.good(), "targetrequest contexts round output failed");

    std::cout << "TARGETREQUEST CONTEXTS PASS contexts=" << context_count
              << " prefill=" << prefix_count << " outputs=" << outputs
              << " scheduler=serial_eager_round_robin\nTARGET_REQUEST_DONE\n";
}
