// Research tool: several agents time-sharing one L0 OSCAR target context.
// Each agent prefills AGENTS_CONTEXT tokens (its own offset into AGENTS_IDS),
// decodes a few tokens and is parked as an exact host state; then the agents
// are resumed round-robin (restore -> AGENTS_DECODE greedy tokens -> park).
// Agent 0's resumed tokens are checked against an uninterrupted reference run.
// Reports park/resume times, decode rate and retained host bytes. Not a test.
#include "exl3/text_model.h"
#include "exl3/branch_reference.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
double ms_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return v ? std::atoi(v) : fallback;
}
}  // namespace

int main() {
    try {
        using namespace ninfer::exl3;
        const char* target = std::getenv("NINFER_EXL3_TARGET_PATH");
        const char* ids_path = std::getenv("AGENTS_IDS");
        if (!target || !ids_path) { std::cerr << "set NINFER_EXL3_TARGET_PATH, AGENTS_IDS\n"; return 77; }
        const int agents = env_int("AGENTS_COUNT", 4);
        const int context = env_int("AGENTS_CONTEXT", 16384);
        const int decode = env_int("AGENTS_DECODE", 32);
        const int rounds = env_int("AGENTS_ROUNDS", 2);
        std::vector<std::int64_t> ids;
        std::ifstream f(ids_path);
        for (std::int64_t t; f >> t;) ids.push_back(t);
        const std::size_t stride = 7919;
        if (ids.size() < static_cast<std::size_t>(context) + stride * (agents - 1) + 1)
            throw std::runtime_error("ids shorter than agents x context");
        auto model = Exl3TextModel::load(target, context + decode * (rounds + 2) + 64);
        auto ctx = model->create_context(true);
        ctx->bind_request_compatibility("l0-agents");
        const auto prefill = [&](int agent) {
            const std::span<const std::int64_t> all(ids.data() + stride * agent, context);
            ctx->reset_for_request("l0-agents");
            const auto initial = static_cast<std::size_t>(Exl3TextContext::layer_major_initial_rows());
            if (initial) ctx->prefill(all.first(initial));
            ctx->append_prefill_layer_major(all.subspan(initial));
        };
        const auto run = [&](std::vector<std::int64_t>& out, int n) {
            for (int i = 0; i < n; ++i) {
                const auto token = exl3_branch_greedy(*ctx);
                out.push_back(token);
                ctx->decode(token);
            }
        };
        // Reference: agent 0 uninterrupted.
        std::vector<std::int64_t> reference;
        prefill(0);
        run(reference, decode * (rounds + 1));
        cudaDeviceSynchronize();

        std::vector<std::shared_ptr<const Exl3ExactHostState>> parked(agents);
        std::vector<std::vector<std::int64_t>> produced(agents);
        double prefill_ms = 0, park_ms = 0, resume_ms = 0, decode_ms = 0, first_ms = 0;
        int parks = 0, resumes = 0, decoded = 0;
        for (int a = 0; a < agents; ++a) {
            auto start = std::chrono::steady_clock::now();
            prefill(a);
            cudaDeviceSynchronize();
            prefill_ms += ms_since(start);
            start = std::chrono::steady_clock::now();
            run(produced[a], decode);
            cudaDeviceSynchronize();
            decode_ms += ms_since(start); decoded += decode;
            start = std::chrono::steady_clock::now();
            parked[a] = ctx->export_exact_host_state();
            park_ms += ms_since(start); ++parks;
        }
        for (int r = 0; r < rounds; ++r)
            for (int a = 0; a < agents; ++a) {
                auto start = std::chrono::steady_clock::now();
                ctx->restore_exact_host_state(*parked[a]);
                cudaDeviceSynchronize();
                resume_ms += ms_since(start); ++resumes;
                start = std::chrono::steady_clock::now();
                run(produced[a], 1);
                cudaDeviceSynchronize();
                first_ms += ms_since(start);
                start = std::chrono::steady_clock::now();
                run(produced[a], decode - 1);
                cudaDeviceSynchronize();
                decode_ms += ms_since(start); decoded += decode - 1;
                start = std::chrono::steady_clock::now();
                parked[a] = ctx->export_exact_host_state();
                park_ms += ms_since(start); ++parks;
            }
        int match = 0;
        for (std::size_t i = 0; i < produced[0].size() && i < reference.size(); ++i) {
            if (produced[0][i] != reference[i]) break;
            ++match;
        }
        std::size_t bytes = 0;
        for (const auto& s : parked) bytes += s->payload_bytes();
        std::cout << "AGENTS count=" << agents << " context=" << context
                  << " combined_tokens=" << static_cast<long long>(agents) * context
                  << " prefill_ms_per_agent=" << prefill_ms / agents
                  << " park_ms=" << park_ms / parks << " resume_ms=" << resume_ms / resumes << " first_token_ms=" << first_ms / resumes
                  << " decode_tok_s=" << decoded / (decode_ms / 1000.0)
                  << " agent0_match=" << match << "/" << produced[0].size()
                  << " parked_GiB=" << bytes / double(1ull << 30) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "AGENTS FAIL: " << e.what() << '\n';
        return 1;
    }
}
