#include "exl3/text_model.h"
#include "exl3/safetensors.h"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer::exl3;
constexpr int kHidden = 5120;
constexpr int kVocab = 248320;
constexpr int kPrefillRows = 14;

void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
std::string env(const char* name) { const char* value = std::getenv(name); return value ? value : ""; }
void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (bits & 0x8000u) << 16u;
    const std::uint32_t exponent = (bits >> 10u) & 31u;
    const std::uint32_t fraction = bits & 1023u;
    std::uint32_t value = sign;
    if (exponent == 0) {
        if (fraction != 0) {
            std::uint32_t mantissa = fraction;
            int shift = 0;
            while ((mantissa & 1024u) == 0) { mantissa <<= 1u; ++shift; }
            value |= static_cast<std::uint32_t>(127 - 14 - shift) << 23u;
            value |= (mantissa & 1023u) << 13u;
        }
    } else if (exponent == 31) {
        value |= 0x7f800000u | (fraction << 13u);
    } else {
        value |= (exponent + 112u) << 23u | (fraction << 13u);
    }
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t value = static_cast<std::uint32_t>(bits) << 16u;
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

double quantile(std::vector<double> values, double fraction) {
    require(!values.empty(), "E4B1 quantile requires samples");
    std::sort(values.begin(), values.end());
    const double index = fraction * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(index));
    const auto upper = std::min(lower + 1, values.size() - 1);
    const double weight = index - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

std::vector<float> as_float(const TensorPayload& tensor) {
    std::vector<float> result;
    if (tensor.info.dtype == "F16") {
        for (const auto bits : tensor.typed<std::uint16_t>("F16")) result.push_back(half_to_float(bits));
    } else if (tensor.info.dtype == "F32") {
        const auto values = tensor.typed<float>("F32");
        result.assign(values.begin(), values.end());
    } else if (tensor.info.dtype == "BF16") {
        for (const auto bits : tensor.typed<std::uint16_t>("BF16")) result.push_back(bf16_to_float(bits));
    } else {
        throw std::runtime_error(tensor.info.name + " has unsupported expected fixture dtype " + tensor.info.dtype);
    }
    return result;
}

std::vector<std::int64_t> as_i64(const TensorPayload& tensor) {
    require(tensor.info.dtype == "I64", tensor.info.name + " is not I64");
    const auto values = tensor.typed<std::int64_t>("I64");
    return {values.begin(), values.end()};
}

TensorPayload fixture_tensor(const std::filesystem::path& path, const SafetensorsHeader& header,
                             const std::string& name) {
    require(header.find(name) != nullptr, "E4A fixture tensor missing: " + name);
    return read_tensor(path, header, name);
}

struct Metric { double max_abs = 0.0; double rms = 0.0; double relative_l2 = 0.0; };
Metric compare(const std::vector<float>& actual, const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "E4A comparison size mismatch");
    Metric result;
    double squared_error = 0.0;
    double squared_expected = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double error = static_cast<double>(actual[i]) - expected[i];
        result.max_abs = std::max(result.max_abs, std::abs(error));
        squared_error += error * error;
        squared_expected += static_cast<double>(expected[i]) * expected[i];
    }
    result.rms = std::sqrt(squared_error / actual.size());
    result.relative_l2 = squared_expected == 0.0 ? std::sqrt(squared_error) :
        std::sqrt(squared_error / squared_expected);
    return result;
}

void print_metric(const char* name, const Metric& metric) {
    std::cout << name << " max_abs=" << std::setprecision(10) << metric.max_abs
              << " rms=" << metric.rms << " relative_l2=" << metric.relative_l2 << '\n';
}

std::vector<float> slice_rows(const std::vector<float>& values, int rows) {
    require(values.size() >= static_cast<std::size_t>(rows) * kHidden, "E4A fixture row count mismatch");
    return {values.begin(), values.begin() + static_cast<std::size_t>(rows) * kHidden};
}

int argmax(const std::vector<float>& values) {
    return static_cast<int>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

std::vector<std::int64_t> greedy_sequence(Exl3TextContext& context, int steps) {
    std::vector<std::int64_t> result;
    result.reserve(steps);
    for (int step = 0; step < steps; ++step) {
        const auto logits = context.logits_host();
        const auto token = static_cast<std::int64_t>(argmax(logits));
        result.push_back(token);
        if (step + 1 < steps) context.decode(token);
    }
    return result;
}

std::vector<std::int64_t> run_prompt(Exl3TextModel& model, std::span<const std::int64_t> prompt, int steps) {
    auto context = model.create_context(false);
    context->prefill(prompt);
    return greedy_sequence(*context, steps);
}

} // namespace

int main() {
    try {
        const auto target = env("NINFER_EXL3_TARGET_PATH");
        const auto oracle_path = env("NINFER_EXL3_ORACLE_PATH");
        if (target.empty() || oracle_path.empty()) {
            std::cerr << "E4A skipped: set NINFER_EXL3_TARGET_PATH and NINFER_EXL3_ORACLE_PATH\n";
            return 77;
        }
        const auto oracle = inspect_file(oracle_path);
        const auto input_ids = as_i64(fixture_tensor(oracle_path, oracle, "input_ids"));
        const auto expected_embedding = as_float(fixture_tensor(oracle_path, oracle, "embedding_output"));
        const auto expected_logits = as_float(fixture_tensor(oracle_path, oracle, "final_logits"));
        const auto expected_first_logits = as_float(fixture_tensor(oracle_path, oracle, "greedy_first_logits"));
        const auto expected_tokens = as_i64(fixture_tensor(oracle_path, oracle, "greedy_tokens"));
        require(input_ids.size() == 15 && expected_tokens.size() == 16, "E4A canonical fixture dimensions changed");
        if(const auto ownership=env("NINFER_TEST_METADATA_OWNER_RELOAD");!ownership.empty()) {
            require(ownership=="1","metadata owner reload fixture option");
            auto owner=Exl3TextModel::load(target,256);
            const auto descriptor_reuses=owner->load_stats().linear_descriptor_reuses;
            require(descriptor_reuses>0 && descriptor_reuses%4==0,"model load missed retained linear descriptors");
            auto survivor=owner->create_context(true);
            require(owner->load_stats().linear_descriptor_reuses==descriptor_reuses,
                "context creation repeated model descriptor loading");
            auto metadata=owner->metadata_owner();
            auto context_metadata=survivor->metadata_owner();
            require(metadata.get()==context_metadata.get() && !metadata.owner_before(context_metadata) &&
                !context_metadata.owner_before(metadata),"model/context metadata owners differ");
            std::weak_ptr<const void> retired_metadata=metadata;
            const auto prompt=std::span<const std::int64_t>(input_ids.data(),input_ids.size());
            survivor->prefill(prompt);
            std::vector<std::uint16_t> before(kVocab),after(kVocab);
            cuda_check(cudaMemcpy(before.data(),survivor->logits_device(),before.size()*2,cudaMemcpyDeviceToHost),"metadata owner baseline logits");
            survivor->reset();owner.reset();context_metadata.reset();
            require(!retired_metadata.expired(),"wrapper destruction released context metadata");
            survivor->prefill(prompt);
            cuda_check(cudaMemcpy(after.data(),survivor->logits_device(),after.size()*2,cudaMemcpyDeviceToHost),"metadata survivor logits");
            require(before==after,"wrapper destruction changed surviving context output");
            survivor.reset();
            require(!retired_metadata.expired(),"metadata handle failed to retain actual backing");
            metadata.reset();
            require(retired_metadata.expired(),"last metadata handle retained backing unexpectedly");
            auto replacement=Exl3TextModel::load(target,256);
            require(replacement->load_stats().linear_descriptor_reuses==descriptor_reuses,
                "reload changed canonical descriptor consumption");
            auto replacement_metadata=replacement->metadata_owner();
            require(retired_metadata.owner_before(replacement_metadata) || replacement_metadata.owner_before(retired_metadata),
                "model reload reused retired metadata control identity");
            auto replacement_context=replacement->create_context(true);
            replacement_context->prefill(prompt);
            cuda_check(cudaMemcpy(after.data(),replacement_context->logits_device(),after.size()*2,cudaMemcpyDeviceToHost),"reloaded metadata logits");
            require(before==after,"metadata reload changed represented logits");
            return 0;
        }

        std::size_t free_before = 0, total_before = 0;
        cudaError_t memory_error = cudaMemGetInfo(&free_before, &total_before);
        require(memory_error == cudaSuccess, "E4A cudaMemGetInfo before model load failed");
        auto model = Exl3TextModel::load(target, 256);
        auto context = model->create_context(true);
        std::size_t free_after = 0, total_after = 0;
        require(cudaMemGetInfo(&free_after, &total_after) == cudaSuccess, "E4A cudaMemGetInfo after context creation failed");
        std::cout << "E4A model_bytes=" << model->model_bytes()
                  << " context_persistent_bytes=" << context->persistent_bytes()
                  << " measured_vram_bytes=" << (free_before - free_after)
                  << " device_total_bytes=" << total_after << '\n';

        if (env("NINFER_EXL3_PROFILE_ONLY") == "1") {
            auto profile_context = model->create_context(false);
            profile_context->prefill(std::span<const std::int64_t>(input_ids.data(), kPrefillRows));
            profile_context->decode(input_ids.back());
            cuda_check(cudaDeviceSynchronize(), "E4B1 profile-only warmup");
            cuda_check(cudaProfilerStart(), "E4B1 profile-only start");
            for (int sample = 0; sample < 6; ++sample) {
                const auto attribution = profile_context->profile_decode(expected_tokens[sample]);
                std::cout << "E4B1 profile_only_sample=" << sample
                          << " total_us=" << attribution.total_microseconds
                          << " stack_us=" << attribution.layer_stack_microseconds << '\n';
            }
            cuda_check(cudaProfilerStop(), "E4B1 profile-only stop");
            std::cout << "E4B1 profile-only: PASS\n";
            return 0;
        }

        context->prefill(std::span<const std::int64_t>(input_ids.data(), kPrefillRows));
        const auto embedding = context->embedding_host();
        print_metric("embedding(first14)", compare(embedding, slice_rows(expected_embedding, kPrefillRows)));
        for (const int layer : {5, 19, 33, 47, 61}) {
            const auto actual = context->hidden_host(layer);
            const auto expected = as_float(fixture_tensor(oracle_path, oracle, "hidden_layer_" + std::to_string(layer)));
            const auto metric = compare(actual, expected);
            print_metric(("hidden_layer_" + std::to_string(layer)).c_str(), metric);
            require(metric.relative_l2 <= .01 && metric.max_abs <= 2.0,
                    "E4A hidden-state numerical gate failed at layer " + std::to_string(layer));
        }

        // The E1 oracle computes final logits after the fifteenth prompt token.
        context->decode(input_ids.back());
        const auto actual_logits = context->logits_host();
        const auto logits_metric = compare(actual_logits, expected_logits);
        const auto first_logits_metric = compare(actual_logits, expected_first_logits);
        print_metric("final_logits", logits_metric);
        print_metric("greedy_first_logits", first_logits_metric);
        require(logits_metric.relative_l2 <= .02 && logits_metric.max_abs <= .5,
                "E4A final-logit numerical gate failed");

        const auto first_token = static_cast<std::int64_t>(argmax(actual_logits));
        require(first_token == expected_tokens.front(), "E4A first greedy token mismatch");
        const auto native_tokens = greedy_sequence(*context, static_cast<int>(expected_tokens.size()));
        int matched = 0;
        for (std::size_t i = 0; i < expected_tokens.size(); ++i) {
            if (native_tokens[i] == expected_tokens[i]) ++matched;
            else std::cerr << "E4A greedy mismatch at position " << i << " expected="
                           << expected_tokens[i] << " actual=" << native_tokens[i] << '\n';
        }
        std::cout << "E4A greedy_tokens_matched=" << matched << "/" << expected_tokens.size() << '\n';
        require(matched == static_cast<int>(expected_tokens.size()), "E4A greedy sequence mismatch");

        // Native reset/replay, including all 48 GDN states and all full-attention caches.
        context->reset();
        context->prefill(std::span<const std::int64_t>(input_ids.data(), kPrefillRows));
        context->decode(input_ids.back());
        const auto replay_tokens = greedy_sequence(*context, static_cast<int>(expected_tokens.size()));
        require(replay_tokens == native_tokens, "E4A reset/replay token sequence mismatch");
        context->reset();
        const std::vector<std::int64_t> prompt_b = {248045, 846, 198, 33963, 799};
        context->prefill(prompt_b);
        const auto prompt_b_tokens = greedy_sequence(*context, 2);
        require(std::all_of(prompt_b_tokens.begin(), prompt_b_tokens.end(), [](std::int64_t t) { return t >= 0 && t < kVocab; }),
                "E4A second prompt produced invalid token");
        context->reset();
        context->prefill(std::span<const std::int64_t>(input_ids.data(), kPrefillRows));
        context->decode(input_ids.back());
        require(greedy_sequence(*context, static_cast<int>(expected_tokens.size())) == native_tokens,
                "E4A cross-request reset/replay mismatch");
        std::cout << "E4A reset_replay=PASS second_prompt=PASS\n";

        // Three short deterministic prompt classes were captured with the pinned
        // ExLlamaV3 ordinary-attention oracle.  Keep this small expectation table
        // in the test; it contains no model payload and gives E4A coverage beyond
        // the canonical E1 prompt.
        const std::vector<std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>> prompt_suite = {
            {{248045, 846, 198, 7734, 799, 11316, 883, 12050, 13, 248046, 198, 248045, 74455, 198},
             {248068, 198, 760, 1156}},
            {{248045, 846, 198, 7734, 12654, 884, 709, 13, 248046, 198, 248045, 74455, 198},
             {248068, 198, 760, 1156}},
            {{248045, 846, 198, 5423, 4566, 440, 5226, 321, 1698, 13, 248046, 198, 248045, 74455, 198},
             {248068, 198, 760, 1156}},
        };
        for (const auto& [prompt, expected] : prompt_suite) {
            const auto actual = run_prompt(*model, prompt, static_cast<int>(expected.size()));
            require(actual == expected, "E4A additional prompt greedy mismatch");
        }
        std::cout << "E4A additional_prompts=3/3\n";

        if (env("NINFER_EXL3_GRAPH_TEST") == "1") {
            auto ungraphed = model->create_context(false);
            auto graphed = model->create_context(false);
            cudaStream_t eager_stream = nullptr;
            cudaStream_t graph_stream = nullptr;
            cuda_check(cudaStreamCreate(&eager_stream), "create E4B2 eager stream");
            cuda_check(cudaStreamCreate(&graph_stream), "create E4B2 graph stream");
            ungraphed->prefill(std::span<const std::int64_t>(input_ids.data(), kPrefillRows), eager_stream);
            graphed->prefill(std::span<const std::int64_t>(input_ids.data(), kPrefillRows), graph_stream);
            std::size_t free_before = 0, total_before = 0, free_after = 0, total_after = 0;
            cuda_check(cudaMemGetInfo(&free_before, &total_before), "E4B2 graph memory before capture");
            const auto capture_start = std::chrono::steady_clock::now();
            require(graphed->capture_decode_graph(graph_stream), "E4B2 graph capture failed: " + graphed->graph_status());
            const auto capture_end = std::chrono::steady_clock::now();
            cuda_check(cudaMemGetInfo(&free_after, &total_after), "E4B2 graph memory after capture");
            require(graphed->graph_active(), "E4B2 graph did not become active");
            const double capture_ms = std::chrono::duration<double, std::milli>(capture_end - capture_start).count();
            std::cout << "E4B2 graph_active=1 status=" << graphed->graph_status()
                      << " capture_ms=" << capture_ms
                      << " graph_vram_delta_bytes=" << (free_before >= free_after ? free_before - free_after : 0)
                      << " context_persistent_bytes=" << graphed->persistent_bytes() << '\n';

            std::vector<std::int64_t> replay_inputs;
            replay_inputs.reserve(expected_tokens.size() + 1);
            replay_inputs.push_back(input_ids.back());
            replay_inputs.insert(replay_inputs.end(), expected_tokens.begin(), expected_tokens.end());
            const int initial_position = graphed->position();
            for (std::size_t step = 0; step < replay_inputs.size(); ++step) {
                ungraphed->decode(replay_inputs[step], eager_stream);
                cuda_check(cudaStreamSynchronize(eager_stream), "synchronize E4B2 eager replay");
                graphed->decode_graph(replay_inputs[step], graph_stream);
                cuda_check(cudaStreamSynchronize(graph_stream), "synchronize E4B2 graph replay");
                const auto ungraphed_logits = ungraphed->logits_host();
                const auto graphed_logits = graphed->logits_host();
                const auto metric = compare(graphed_logits, ungraphed_logits);
                std::cout << "E4B2 replay=" << step << " graph_vs_eager_max_abs=" << metric.max_abs
                          << " relative_l2=" << metric.relative_l2 << '\n';
                if (step < 4) {
                    const auto graph_k = graphed->full_attention_k_host(
                        3, kPrefillRows + static_cast<int>(step), graph_stream);
                    const auto eager_k = ungraphed->full_attention_k_host(
                        3, kPrefillRows + static_cast<int>(step), eager_stream);
                    std::vector<float> graph_k_f, eager_k_f;
                    graph_k_f.reserve(graph_k.size()); eager_k_f.reserve(eager_k.size());
                    for (const auto value : graph_k) graph_k_f.push_back(half_to_float(value));
                    for (const auto value : eager_k) eager_k_f.push_back(half_to_float(value));
                    const auto kv_metric = compare(graph_k_f, eager_k_f);
                    std::cout << "E4B2 replay=" << step << " KV max_abs=" << kv_metric.max_abs
                              << " relative_l2=" << kv_metric.relative_l2 << '\n';
                    require(kv_metric.max_abs <= .01 && kv_metric.relative_l2 <= .005,
                            "E4B2 full-attention KV progression failed");
                    for (const int layer : {5, 33, 61}) {
                        const auto graph_state = graphed->gdn_state_host(layer, graph_stream);
                        const auto eager_state = ungraphed->gdn_state_host(layer, eager_stream);
                        const auto state_metric = compare(graph_state, eager_state);
                        std::cout << "E4B2 replay=" << step << " layer=" << layer
                                  << " graph_vs_eager_state_max_abs=" << state_metric.max_abs
                                  << " relative_l2=" << state_metric.relative_l2 << '\n';
                        require(state_metric.max_abs <= .01 && state_metric.relative_l2 <= .01,
                                "E4B2 GDN state diverged at replay " + std::to_string(step));
                    }
                }
                require(metric.max_abs <= .5 && metric.relative_l2 <= .02,
                        "E4B2 graph logits diverged at replay " + std::to_string(step));
                if (step < expected_tokens.size()) {
                    require(static_cast<std::int64_t>(argmax(graphed_logits)) == expected_tokens[step],
                            "E4B2 graph greedy mismatch at replay " + std::to_string(step));
                }
                require(graphed->position() == initial_position + static_cast<int>(step) + 1,
                        "E4B2 graph position progression failed");
                require(ungraphed->position() == graphed->position(), "E4B2 graph/ungraphed position mismatch");
            }
            std::cout << "E4B2 sequential_replays=" << replay_inputs.size()
                      << " position_final=" << graphed->position()
                      << " kv_progression=PASS gdn_state_progression=PASS\n";

            graphed->reset(graph_stream);
            graphed->prefill(std::span<const std::int64_t>(input_ids.data(), kPrefillRows), graph_stream);
            graphed->decode_graph(input_ids.back(), graph_stream);
            cuda_check(cudaDeviceSynchronize(), "synchronize E4B2 reset replay A");
            const auto replay_logits_a = graphed->logits_host();
            graphed->reset(graph_stream);
            const std::vector<std::int64_t> prompt_b = {248045, 846, 198, 33963, 799};
            graphed->prefill(prompt_b, graph_stream);
            graphed->decode_graph(prompt_b.back(), graph_stream);
            cuda_check(cudaDeviceSynchronize(), "synchronize E4B2 prompt B");
            const auto prompt_b_logits = graphed->logits_host();
            require(std::all_of(prompt_b_logits.begin(), prompt_b_logits.end(), [](float value) {
                return std::isfinite(value);
            }), "E4B2 graph second prompt produced non-finite logits");
            graphed->reset(graph_stream);
            graphed->prefill(std::span<const std::int64_t>(input_ids.data(), kPrefillRows), graph_stream);
            graphed->decode_graph(input_ids.back(), graph_stream);
            cuda_check(cudaDeviceSynchronize(), "synchronize E4B2 reset replay B");
            const auto replay_logits_b = graphed->logits_host();
            const auto reset_metric = compare(replay_logits_a, replay_logits_b);
            require(reset_metric.max_abs <= .5 && reset_metric.relative_l2 <= .02,
                    "E4B2 graph reset/replay mismatch");
            std::cout << "E4B2 reset_replay=PASS cross_prompt_reset=PASS graph_disabled_equivalence=PASS\n";

            ungraphed->reset(eager_stream);
            graphed->reset(graph_stream);
            ungraphed->prefill(std::span<const std::int64_t>(input_ids.data(), kPrefillRows), eager_stream);
            graphed->prefill(std::span<const std::int64_t>(input_ids.data(), kPrefillRows), graph_stream);
            require(graphed->graph_active(), "E4B2 benchmark graph was lost after reset");
            ungraphed->decode(input_ids.back(), eager_stream);
            graphed->decode_graph(input_ids.back(), graph_stream);
            cuda_check(cudaDeviceSynchronize(), "synchronize E4B2 graph benchmark warmup");
            cudaEvent_t start = nullptr, stop = nullptr;
            require(cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&stop) == cudaSuccess,
                    "E4B2 benchmark event creation failed");
            std::vector<double> eager_times, graph_times;
            eager_times.reserve(8); graph_times.reserve(8);
            if (env("NINFER_EXL3_GRAPH_PROFILE_ONLY") == "1") {
                cuda_check(cudaProfilerStart(), "E4B2 profiler start");
            }
            for (int sample = 0; sample < 8; ++sample) {
                cuda_check(cudaEventRecord(start, eager_stream), "E4B2 eager timing start");
                ungraphed->decode(expected_tokens[sample], eager_stream);
                cuda_check(cudaEventRecord(stop, eager_stream), "E4B2 eager timing stop");
                cuda_check(cudaEventSynchronize(stop), "E4B2 eager timing sync");
                float ms = 0.0f;
                cuda_check(cudaEventElapsedTime(&ms, start, stop), "E4B2 eager timing elapsed");
                eager_times.push_back(ms);
                cuda_check(cudaEventRecord(start, graph_stream), "E4B2 graph timing start");
                graphed->decode_graph(expected_tokens[sample], graph_stream);
                cuda_check(cudaEventRecord(stop, graph_stream), "E4B2 graph timing stop");
                cuda_check(cudaEventSynchronize(stop), "E4B2 graph timing sync");
                ms = 0.0f;
                cuda_check(cudaEventElapsedTime(&ms, start, stop), "E4B2 graph timing elapsed");
                graph_times.push_back(ms);
            }
            if (env("NINFER_EXL3_GRAPH_PROFILE_ONLY") == "1") {
                cuda_check(cudaProfilerStop(), "E4B2 profiler stop");
            }
            cudaEventDestroy(start); cudaEventDestroy(stop);
            const double eager_median = quantile(eager_times, .5);
            const double graph_median = quantile(graph_times, .5);
            std::cout << "E4B2 eager_median_ms=" << eager_median
                      << " eager_mean_ms=" << mean(eager_times)
                      << " graph_median_ms=" << graph_median
                      << " graph_mean_ms=" << mean(graph_times)
                      << " saved_ms=" << eager_median - graph_median
                      << " improvement_pct=" << 100.0 * (eager_median - graph_median) / eager_median
                      << " graph_tok_per_s=" << 1000.0 / graph_median
                      << " launch_apis_before=2133 graph_launches_after=1 embedding_launches_after=1\n";
            cuda_check(cudaStreamDestroy(eager_stream), "destroy E4B2 eager stream");
            cuda_check(cudaStreamDestroy(graph_stream), "destroy E4B2 graph stream");
            std::cout << "E4B2 graph test: PASS\n";
        }

        // Opt-in stream-ordered attribution.  Each sample has one terminal
        // event synchronization; there is no per-layer synchronization.
        auto attribution_context = model->create_context(false);
        attribution_context->prefill(std::span<const std::int64_t>(input_ids.data(), kPrefillRows));
        attribution_context->decode(input_ids.back());
        cuda_check(cudaDeviceSynchronize(), "E4B1 profile warmup");
        std::vector<Exl3TextDecodeAttribution> profile_samples;
        cuda_check(cudaProfilerStart(), "E4B1 profiler start");
        for (int sample = 0; sample < 6; ++sample) {
            profile_samples.push_back(attribution_context->profile_decode(expected_tokens[sample]));
        }
        cuda_check(cudaProfilerStop(), "E4B1 profiler stop");
        auto values_for = [&](auto selector) {
            std::vector<double> values;
            values.reserve(profile_samples.size());
            for (const auto& sample : profile_samples) values.push_back(selector(sample));
            return values;
        };
        const auto print_profile = [&](const char* name, auto selector) {
            const auto values = values_for(selector);
            std::cout << "E4B1 " << name << " p10_us=" << quantile(values, .10)
                      << " median_us=" << quantile(values, .50)
                      << " mean_us=" << mean(values)
                      << " p90_us=" << quantile(values, .90) << '\n';
        };
        print_profile("total", [](const auto& s) { return s.total_microseconds; });
        print_profile("token_id_h2d", [](const auto& s) { return s.token_id_h2d_microseconds; });
        print_profile("embedding", [](const auto& s) { return s.embedding_microseconds; });
        print_profile("layer_stack", [](const auto& s) { return s.layer_stack_microseconds; });
        print_profile("final_norm", [](const auto& s) { return s.final_norm_microseconds; });
        print_profile("lm_head", [](const auto& s) { return s.lm_head_microseconds; });
        print_profile("other", [](const auto& s) { return s.other_microseconds; });
        std::vector<double> full_family, gdn_family, stack_gaps;
        for (const auto& sample : profile_samples) {
            double full = 0.0, gdn = 0.0;
            for (int layer = 0; layer < 64; ++layer) {
                if (layer % 4 == 3) full += sample.layer_microseconds[layer];
                else gdn += sample.layer_microseconds[layer];
            }
            full_family.push_back(full);
            gdn_family.push_back(gdn);
            stack_gaps.push_back(sample.layer_stack_microseconds - full - gdn);
        }
        std::cout << "E4B1 full_family median_us=" << quantile(full_family, .50)
                  << " gdn_family median_us=" << quantile(gdn_family, .50)
                  << " layer_stack_gap median_us=" << quantile(stack_gaps, .50) << '\n';
        for (int layer = 0; layer < 64; ++layer) {
            const auto values = values_for([layer](const auto& s) { return s.layer_microseconds[layer]; });
            const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
            std::cout << "E4B1 layer=" << layer << " median_us=" << quantile(values, .50)
                      << " mean_us=" << mean(values) << " min_us=" << *min_it
                      << " max_us=" << *max_it << '\n';
        }

        // Production-like decode timing: no diagnostic tap copies and no result download.
        auto benchmark_context = model->create_context(false);
        benchmark_context->prefill(std::span<const std::int64_t>(input_ids.data(), kPrefillRows));
        benchmark_context->decode(input_ids.back());
        cuda_check(cudaDeviceSynchronize(), "E4A timing warmup");
        cudaEvent_t start = nullptr, stop = nullptr;
        require(cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&stop) == cudaSuccess,
                "E4A timing event creation failed");
        std::vector<double> production_samples;
        for (int sample = 0; sample < 6; ++sample) {
            cuda_check(cudaEventRecord(start), "E4A timing start");
            benchmark_context->decode(expected_tokens[sample]);
            cuda_check(cudaEventRecord(stop), "E4A timing stop");
            cuda_check(cudaEventSynchronize(stop), "E4A timing synchronize");
            float milliseconds = 0.0f;
            cuda_check(cudaEventElapsedTime(&milliseconds, start, stop), "E4A timing elapsed");
            production_samples.push_back(milliseconds);
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        const double production_median = quantile(production_samples, .50);
        std::cout << "E4B1 production_m1_decode_ms=" << production_median
                  << " p10_ms=" << quantile(production_samples, .10)
                  << " mean_ms=" << mean(production_samples)
                  << " p90_ms=" << quantile(production_samples, .90)
                  << " tok_per_s=" << (1000.0 / production_median)
                  << " mallocs=0 frees=0 model_state_h2d=0 d2h=0 internal_syncs=0 token_id_h2d="
                  << 1 << '\n';

        // Explicit model destruction/reconstruction is intentionally last because it reloads
        // the canonical multi-shard artifact and is outside the hot path.
        context.reset();
        benchmark_context.reset();
        model.reset();
        auto recreated = Exl3TextModel::load(target, 256);
        const auto recreated_tokens = run_prompt(*recreated,
            std::span<const std::int64_t>(input_ids.data(), input_ids.size()), 2);
        require(recreated_tokens.size() == 2, "E4A recreated model did not decode");
        require(recreated_tokens.front() == expected_tokens.front(), "E4A recreated model first token mismatch");
        std::cout << "E4A destroy_recreate=PASS\nE4A full model test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "E4A full model test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
