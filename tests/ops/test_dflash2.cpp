#include "ninfer/ops/dflash2_dynamic_conv.h"
#include "ninfer/ops/dflash2_qkv_proj.h"
#include "ninfer/ops/dflash2_predecessor_ids.h"
#include "ninfer/ops/dflash2_selector_lattice.h"
#include "ninfer/ops/linear.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::test;

namespace {

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

int run_dynamic_conv_case(int side) {
    constexpr std::int32_t hidden = 32;
    constexpr std::int32_t tokens = 8;
    constexpr std::int32_t width  = 4;
    constexpr std::int32_t groups = hidden / kDFlash2ConvGroupSize;
    constexpr std::int32_t dyn_rows = kDFlash2ConvKernel * kDFlash2ConvKernel * groups;

    std::vector<float> x(static_cast<std::size_t>(hidden) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t c = 0; c < hidden; ++c) {
            x[static_cast<std::size_t>(t) * hidden + c] =
                1.0f + 0.125f * static_cast<float>(c) + 0.25f * static_cast<float>(t);
        }
    }
    round_to_bf16(x);

    std::vector<float> dynamic(static_cast<std::size_t>(dyn_rows) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t row = 0; row < dyn_rows; ++row) {
            dynamic[static_cast<std::size_t>(t) * dyn_rows + row] =
                0.25f * static_cast<float>(row + 1) + 0.0625f * static_cast<float>(t);
        }
    }
    round_to_bf16(dynamic);

    std::vector<float> base(static_cast<std::size_t>(kDFlash2ConvKernel) *
                            kDFlash2ConvKernel * hidden);
    for (std::int32_t side_index = 0; side_index < kDFlash2ConvKernel; ++side_index) {
        for (std::int32_t tap = 0; tap < kDFlash2ConvKernel; ++tap) {
            for (std::int32_t c = 0; c < hidden; ++c) {
                base[(static_cast<std::size_t>(side_index) * kDFlash2ConvKernel + tap) * hidden +
                     c] = 0.5f * static_cast<float>(side_index + 1) +
                         0.125f * static_cast<float>(tap) + 0.03125f * static_cast<float>(c);
            }
        }
    }
    round_to_bf16(base);

    std::vector<std::uint16_t> expected(static_cast<std::size_t>(hidden) * tokens);
    const std::vector<std::uint16_t> x_bits       = bf16_bits(x);
    const std::vector<std::uint16_t> dynamic_bits = bf16_bits(dynamic);
    const std::vector<std::uint16_t> base_bits    = bf16_bits(base);
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::int32_t position = t % width;
        for (std::int32_t c = 0; c < hidden; ++c) {
            const std::int32_t group = c / kDFlash2ConvGroupSize;
            const std::int32_t row0 = side * kDFlash2ConvKernel * groups;
            const std::int32_t row1 = row0 + groups;
            const float x0 = bf16_to_f32(x_bits[static_cast<std::size_t>(t) * hidden + c]);
            const float x1 = position == 0
                                 ? 0.0f
                                 : bf16_to_f32(x_bits[static_cast<std::size_t>(t - 1) * hidden + c]);
            const float w0 = bf16_to_f32(dynamic_bits[static_cast<std::size_t>(t) * dyn_rows +
                                                       row0 + group]) +
                             bf16_to_f32(base_bits[(static_cast<std::size_t>(side) * 2) * hidden + c]);
            const float w1 = bf16_to_f32(dynamic_bits[static_cast<std::size_t>(t) * dyn_rows +
                                                       row1 + group]) +
                             bf16_to_f32(base_bits[(static_cast<std::size_t>(side) * 2 + 1) * hidden + c]);
            expected[static_cast<std::size_t>(t) * hidden + c] = f32_to_bf16(w0 * x0 + w1 * x1);
        }
    }

    GuardedDeviceBuffer device_x(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_dynamic(dynamic_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_base(base_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(expected.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(x_bits.data(), device_x.bytes());
    device_dynamic.copy_from_host(dynamic_bits.data(), device_dynamic.bytes());
    device_base.copy_from_host(base_bits.data(), device_base.bytes());
    device_out.fill(0xcd);

    Tensor x_tensor(device_x.data(), DType::BF16, {hidden, tokens});
    Tensor dynamic_tensor(device_dynamic.data(), DType::BF16, {dyn_rows, tokens});
    Tensor base_tensor(device_base.data(), DType::BF16,
                       {kDFlash2ConvKernel, kDFlash2ConvKernel, hidden});
    Tensor out_tensor(device_out.data(), DType::BF16, {hidden, tokens});
    ops::dflash2_dynamic_conv(x_tensor, dynamic_tensor, base_tensor, side, width, out_tensor,
                              nullptr);
    cuda_synchronize();

    const std::string label = "dflash2_dynamic_conv side=" + std::to_string(side);
    int failures = verify_exact(label.c_str(),
                                from_device<std::uint16_t>(device_out.data(), expected.size()),
                                expected);
    failures += verify_exact((label + " x preserved").c_str(),
                             from_device<std::uint16_t>(device_x.data(), x_bits.size()), x_bits);
    failures += verify_exact((label + " dynamic preserved").c_str(),
                             from_device<std::uint16_t>(device_dynamic.data(), dynamic_bits.size()),
                             dynamic_bits);
    failures += verify_exact((label + " base preserved").c_str(),
                             from_device<std::uint16_t>(device_base.data(), base_bits.size()),
                             base_bits);
    failures += device_x.verify_guards(label + " x");
    failures += device_dynamic.verify_guards(label + " dynamic");
    failures += device_base.verify_guards(label + " base");
    failures += device_out.verify_guards(label + " output");
    return failures;
}

int run_dynamic_conv_benchmark() {
    constexpr std::int32_t hidden = 5120;
    constexpr std::int32_t tokens = 8;
    constexpr std::int32_t width = 8;
    constexpr int warmup = 100;
    constexpr int repeats = 2000;
    constexpr std::int32_t dyn_rows =
        kDFlash2ConvKernel * kDFlash2ConvKernel * (hidden / kDFlash2ConvGroupSize);

    const std::vector<std::uint16_t> x(static_cast<std::size_t>(hidden) * tokens,
                                       f32_to_bf16(0.25f));
    const std::vector<std::uint16_t> dynamic(static_cast<std::size_t>(dyn_rows) * tokens,
                                             f32_to_bf16(0.125f));
    const std::vector<std::uint16_t> base(static_cast<std::size_t>(kDFlash2ConvKernel) *
                                              kDFlash2ConvKernel * hidden,
                                          f32_to_bf16(0.0625f));
    GuardedDeviceBuffer device_x(x.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_dynamic(dynamic.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_base(base.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(x.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(x.data(), device_x.bytes());
    device_dynamic.copy_from_host(dynamic.data(), device_dynamic.bytes());
    device_base.copy_from_host(base.data(), device_base.bytes());

    Tensor x_tensor(device_x.data(), DType::BF16, {hidden, tokens});
    Tensor dynamic_tensor(device_dynamic.data(), DType::BF16, {dyn_rows, tokens});
    Tensor base_tensor(device_base.data(), DType::BF16,
                       {kDFlash2ConvKernel, kDFlash2ConvKernel, hidden});
    Tensor out_tensor(device_out.data(), DType::BF16, {hidden, tokens});
    for (int i = 0; i < warmup; ++i) {
        ops::dflash2_dynamic_conv(x_tensor, dynamic_tensor, base_tensor, 0, width, out_tensor,
                                  nullptr);
    }
    cuda_synchronize();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate benchmark start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate benchmark stop");
    std::vector<float> samples;
    samples.reserve(5);
    for (int sample = 0; sample < 5; ++sample) {
        cuda_check(cudaEventRecord(start), "cudaEventRecord benchmark start");
        for (int i = 0; i < repeats; ++i) {
            ops::dflash2_dynamic_conv(x_tensor, dynamic_tensor, base_tensor, 0, width, out_tensor,
                                      nullptr);
        }
        cuda_check(cudaEventRecord(stop), "cudaEventRecord benchmark stop");
        cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize benchmark stop");
        float elapsed_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop),
                   "cudaEventElapsedTime benchmark");
        samples.push_back(elapsed_ms * 1000.0f / repeats);
    }
    std::sort(samples.begin(), samples.end());
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy benchmark start");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy benchmark stop");
    std::cout << "dflash2_dynamic_conv benchmark hidden=" << hidden << " tokens=" << tokens
              << " us_median=" << samples[samples.size() / 2] << " us_min=" << samples.front()
              << " us_max=" << samples.back() << '\n';
    return 0;
}

int run_qkv_projection_case() {
    constexpr std::int32_t input_rows = 5120;
    constexpr std::int32_t query_rows = 4096;
    constexpr std::int32_t kv_rows = 1024;
    constexpr std::int32_t parent_rows = query_rows + 2 * kv_rows;
    constexpr std::int32_t columns = 8;
    constexpr std::int32_t groups = input_rows / 32;

    std::vector<std::uint8_t> codes(static_cast<std::size_t>(parent_rows) * input_rows);
    for (std::int32_t row = 0; row < parent_rows; ++row) {
        for (std::int32_t col = 0; col < input_rows; ++col) {
            const std::int32_t value = ((row * 13 + col * 7) % 31) - 15;
            codes[static_cast<std::size_t>(row) * input_rows + col] =
                static_cast<std::uint8_t>(static_cast<std::int8_t>(value));
        }
    }
    const std::vector<std::uint16_t> scales(
        static_cast<std::size_t>(parent_rows) * groups, 0x3c00u); // FP16 1.0
    const std::vector<std::uint16_t> x(static_cast<std::size_t>(input_rows) * columns,
                                       f32_to_bf16(0.03125f));

    GuardedDeviceBuffer device_codes(codes.size());
    GuardedDeviceBuffer device_scales(scales.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_x(x.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_packed(static_cast<std::size_t>(parent_rows) * columns *
                                      sizeof(std::uint16_t));
    GuardedDeviceBuffer device_q(static_cast<std::size_t>(query_rows) * columns *
                                  sizeof(std::uint16_t));
    GuardedDeviceBuffer device_k(static_cast<std::size_t>(kv_rows) * columns *
                                  sizeof(std::uint16_t));
    GuardedDeviceBuffer device_v(static_cast<std::size_t>(kv_rows) * columns *
                                  sizeof(std::uint16_t));
    device_codes.copy_from_host(codes.data(), device_codes.bytes());
    device_scales.copy_from_host(scales.data(), device_scales.bytes());
    device_x.copy_from_host(x.data(), device_x.bytes());

    Weight weight{};
    weight.payload = device_codes.data();
    weight.payload_bytes = codes.size() + device_scales.bytes();
    weight.qtype = QType::W8G32_F16S;
    weight.group_size = 32;
    weight.ndim = 2;
    weight.qdata = device_codes.data();
    weight.scales = device_scales.data();
    weight.n = parent_rows;
    weight.k = input_rows;
    weight.group = 32;
    weight.layout = QuantLayout::RowSplit;
    weight.scale_dtype = DType::FP16;
    weight.shape[0] = weight.padded_shape[0] = parent_rows;
    weight.shape[1] = weight.padded_shape[1] = input_rows;

    Tensor x_tensor(device_x.data(), DType::BF16, {input_rows, columns});
    Tensor packed_tensor(device_packed.data(), DType::BF16, {parent_rows, columns});
    Tensor q_tensor(device_q.data(), DType::BF16, {query_rows, columns});
    Tensor k_tensor(device_k.data(), DType::BF16, {kv_rows, columns});
    Tensor v_tensor(device_v.data(), DType::BF16, {kv_rows, columns});
    ops::linear(x_tensor, weight, packed_tensor, nullptr);
    ops::dflash2_qkv_proj(x_tensor, weight, q_tensor, k_tensor, v_tensor, nullptr);
    cuda_synchronize();

    const std::vector<std::uint16_t> packed =
        from_device<std::uint16_t>(device_packed.data(), static_cast<std::size_t>(parent_rows) * columns);
    std::vector<std::uint16_t> expected_q(static_cast<std::size_t>(query_rows) * columns);
    std::vector<std::uint16_t> expected_k(static_cast<std::size_t>(kv_rows) * columns);
    std::vector<std::uint16_t> expected_v(static_cast<std::size_t>(kv_rows) * columns);
    for (std::int32_t t = 0; t < columns; ++t) {
        const std::size_t packed_column = static_cast<std::size_t>(t) * parent_rows;
        std::copy_n(packed.begin() + packed_column, query_rows,
                    expected_q.begin() + static_cast<std::size_t>(t) * query_rows);
        std::copy_n(packed.begin() + packed_column + query_rows, kv_rows,
                    expected_k.begin() + static_cast<std::size_t>(t) * kv_rows);
        std::copy_n(packed.begin() + packed_column + query_rows + kv_rows, kv_rows,
                    expected_v.begin() + static_cast<std::size_t>(t) * kv_rows);
    }

    const std::string label = "dflash2_qkv_proj";
    int failures = verify_exact((label + " q").c_str(),
                                from_device<std::uint16_t>(device_q.data(), expected_q.size()),
                                expected_q);
    failures += verify_exact((label + " k").c_str(),
                             from_device<std::uint16_t>(device_k.data(), expected_k.size()),
                             expected_k);
    failures += verify_exact((label + " v").c_str(),
                             from_device<std::uint16_t>(device_v.data(), expected_v.size()),
                             expected_v);
    failures += device_codes.verify_guards(label + " codes");
    failures += device_scales.verify_guards(label + " scales");
    failures += device_x.verify_guards(label + " x");
    failures += device_packed.verify_guards(label + " packed");
    failures += device_q.verify_guards(label + " q");
    failures += device_k.verify_guards(label + " k");
    failures += device_v.verify_guards(label + " v");

    if (std::getenv("NINFER_DFLASH2_BENCH") != nullptr) {
        constexpr int warmup = 20;
        constexpr int repeats = 100;
        constexpr int samples_count = 5;
        const auto copy_packed = [&] {
            constexpr std::size_t element_bytes = sizeof(std::uint16_t);
            const auto copy_rows = [&](std::int32_t source_row, std::int32_t rows, void* destination) {
                cuda_check(cudaMemcpy2DAsync(
                               destination, static_cast<std::size_t>(rows) * element_bytes,
                               static_cast<const std::byte*>(device_packed.data()) +
                                   static_cast<std::size_t>(source_row) * element_bytes,
                               static_cast<std::size_t>(parent_rows) * element_bytes,
                               static_cast<std::size_t>(rows) * element_bytes,
                               static_cast<std::size_t>(columns), cudaMemcpyDeviceToDevice, nullptr),
                           "cudaMemcpy2DAsync DFlash2 QKV benchmark");
            };
            copy_rows(0, query_rows, device_q.data());
            copy_rows(query_rows, kv_rows, device_k.data());
            copy_rows(query_rows + kv_rows, kv_rows, device_v.data());
        };
        const auto measure = [&](bool direct) {
            for (int i = 0; i < warmup; ++i) {
                if (direct) {
                    ops::dflash2_qkv_proj(x_tensor, weight, q_tensor, k_tensor, v_tensor, nullptr);
                } else {
                    ops::linear(x_tensor, weight, packed_tensor, nullptr);
                    copy_packed();
                }
            }
            cuda_synchronize();
            cudaEvent_t start = nullptr;
            cudaEvent_t stop = nullptr;
            cuda_check(cudaEventCreate(&start), "cudaEventCreate QKV benchmark start");
            cuda_check(cudaEventCreate(&stop), "cudaEventCreate QKV benchmark stop");
            std::vector<float> samples;
            samples.reserve(samples_count);
            for (int sample = 0; sample < samples_count; ++sample) {
                cuda_check(cudaEventRecord(start), "cudaEventRecord QKV benchmark start");
                for (int i = 0; i < repeats; ++i) {
                    if (direct) {
                        ops::dflash2_qkv_proj(x_tensor, weight, q_tensor, k_tensor, v_tensor,
                                              nullptr);
                    } else {
                        ops::linear(x_tensor, weight, packed_tensor, nullptr);
                        copy_packed();
                    }
                }
                cuda_check(cudaEventRecord(stop), "cudaEventRecord QKV benchmark stop");
                cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize QKV benchmark stop");
                float elapsed_ms = 0.0f;
                cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop),
                           "cudaEventElapsedTime QKV benchmark");
                samples.push_back(elapsed_ms * 1000.0f / repeats);
            }
            cuda_check(cudaEventDestroy(start), "cudaEventDestroy QKV benchmark start");
            cuda_check(cudaEventDestroy(stop), "cudaEventDestroy QKV benchmark stop");
            std::sort(samples.begin(), samples.end());
            return samples;
        };
        const std::vector<float> packed_samples = measure(false);
        const std::vector<float> direct_samples = measure(true);
        std::cout << label << " benchmark packed_plus_copy_us_median="
                  << packed_samples[samples_count / 2] << " direct_us_median="
                  << direct_samples[samples_count / 2] << '\n';
    }
    return failures;
}

int run_candidate_selection_benchmark() {
    constexpr std::int32_t vocab = 248320;
    constexpr std::int32_t columns = 8;
    constexpr int warmup = 20;
    constexpr int repeats = 100;
    constexpr int samples_count = 5;

    std::vector<std::uint16_t> logits(static_cast<std::size_t>(vocab) * columns);
    for (std::int32_t t = 0; t < columns; ++t) {
        for (std::int32_t id = 0; id < vocab; ++id) {
            logits[static_cast<std::size_t>(t) * vocab + id] =
                f32_to_bf16(static_cast<float>((id * 17 + t * 13) % 1009) * 0.03125f);
        }
    }
    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_ids(static_cast<std::size_t>(kDFlash2SelectorTopK) * columns *
                                   sizeof(std::int32_t));
    GuardedDeviceBuffer device_values(static_cast<std::size_t>(kDFlash2SelectorTopK) * columns *
                                      sizeof(float));
    device_logits.copy_from_host(logits.data(), device_logits.bytes());
    Tensor logits_tensor(device_logits.data(), DType::BF16, {vocab, columns});
    Tensor ids_tensor(device_ids.data(), DType::I32, {kDFlash2SelectorTopK, columns});
    Tensor values_tensor(device_values.data(), DType::FP32, {kDFlash2SelectorTopK, columns});

    for (int i = 0; i < warmup; ++i) {
        ops::dflash2_select_candidates(logits_tensor, ids_tensor, values_tensor, nullptr);
    }
    cuda_synchronize();
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate selector benchmark start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate selector benchmark stop");
    std::vector<float> samples;
    samples.reserve(samples_count);
    for (int sample = 0; sample < samples_count; ++sample) {
        cuda_check(cudaEventRecord(start), "cudaEventRecord selector benchmark start");
        for (int i = 0; i < repeats; ++i) {
            ops::dflash2_select_candidates(logits_tensor, ids_tensor, values_tensor, nullptr);
        }
        cuda_check(cudaEventRecord(stop), "cudaEventRecord selector benchmark stop");
        cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize selector benchmark stop");
        float elapsed_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop),
                   "cudaEventElapsedTime selector benchmark");
        samples.push_back(elapsed_ms * 1000.0f / repeats);
    }
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy selector benchmark start");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy selector benchmark stop");
    std::sort(samples.begin(), samples.end());
    std::cout << "dflash2_select_candidates benchmark vocab=" << vocab << " tokens=" << columns
              << " us_median=" << samples[samples.size() / 2] << " us_min=" << samples.front()
              << " us_max=" << samples.back() << '\n';
    return 0;
}

int run_predecessor_ids_case() {
    constexpr std::int32_t top_k        = kDFlash2SelectorTopK;
    constexpr std::int32_t block_tokens = 4;
    constexpr std::int32_t batch        = 2;
    constexpr std::int32_t tokens       = block_tokens * batch;

    std::vector<std::int32_t> candidates(static_cast<std::size_t>(top_k) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t k = 0; k < top_k; ++k) {
            candidates[static_cast<std::size_t>(t) * top_k + k] = 1000 + 100 * t + k;
        }
    }
    const std::vector<std::int32_t> anchors{77, 177};
    std::vector<std::int32_t> expected(candidates.size());
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::int32_t anchor = anchors[t / block_tokens];
        for (std::int32_t k = 0; k < top_k; ++k) {
            expected[static_cast<std::size_t>(t) * top_k + k] =
                (t % block_tokens <= 1) ? anchor
                                         : candidates[static_cast<std::size_t>(t - 1) * top_k + k];
        }
    }

    GuardedDeviceBuffer device_candidates(candidates.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_anchors(anchors.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_out(expected.size() * sizeof(std::int32_t));
    device_candidates.copy_from_host(candidates.data(), device_candidates.bytes());
    device_anchors.copy_from_host(anchors.data(), device_anchors.bytes());
    device_out.fill(0xcd);

    Tensor candidate_tensor(device_candidates.data(), DType::I32, {top_k, tokens});
    Tensor anchor_tensor(device_anchors.data(), DType::I32, {batch});
    Tensor out_tensor(device_out.data(), DType::I32, {top_k, tokens});
    ops::dflash2_predecessor_ids(candidate_tensor, anchor_tensor, block_tokens, out_tensor, nullptr);
    cuda_synchronize();

    const std::string label = "dflash2_predecessor_ids";
    int failures = verify_exact(label.c_str(),
                                from_device<std::int32_t>(device_out.data(), expected.size()),
                                expected);
    failures += verify_exact((label + " candidates preserved").c_str(),
                             from_device<std::int32_t>(device_candidates.data(), candidates.size()),
                             candidates);
    failures += verify_exact((label + " anchors preserved").c_str(),
                             from_device<std::int32_t>(device_anchors.data(), anchors.size()), anchors);
    failures += device_candidates.verify_guards(label + " candidates");
    failures += device_anchors.verify_guards(label + " anchors");
    failures += device_out.verify_guards(label + " output");
    return failures;
}

int run_select_candidates_case() {
    constexpr std::int32_t vocab  = 248320;
    constexpr std::int32_t tokens = 3;
    constexpr std::int32_t top_k  = kDFlash2SelectorTopK;

    std::vector<float> logits(static_cast<std::size_t>(vocab) * tokens, -100.0f);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t id = 0; id < vocab; ++id) {
            logits[static_cast<std::size_t>(t) * vocab + id] =
                -20.0f + static_cast<float>((id * 17 + t * 31) % 97) * 0.125f;
        }
        // Force deterministic ties at the top of each column.
        logits[static_cast<std::size_t>(t) * vocab + 3]  = 40.0f;
        logits[static_cast<std::size_t>(t) * vocab + 11] = 40.0f;
        // IDs 3 and 131 are visited by the same selector thread; this also
        // verifies that a private top-k run preserves its lower-id tie order.
        logits[static_cast<std::size_t>(t) * vocab + 131] = 40.0f;
        logits[static_cast<std::size_t>(t) * vocab + 29] = 39.5f;
        logits[static_cast<std::size_t>(t) * vocab + 31] = 39.5f;
    }
    const std::vector<std::uint16_t> input = bf16_bits(logits);

    std::vector<std::int32_t> expected_ids(static_cast<std::size_t>(top_k) * tokens);
    std::vector<float> expected_values(expected_ids.size());
    for (std::int32_t t = 0; t < tokens; ++t) {
        std::vector<std::int32_t> ids(vocab);
        std::iota(ids.begin(), ids.end(), 0);
        std::sort(ids.begin(), ids.end(), [&](std::int32_t a, std::int32_t b) {
            const float va = bf16_to_f32(input[static_cast<std::size_t>(t) * vocab + a]);
            const float vb = bf16_to_f32(input[static_cast<std::size_t>(t) * vocab + b]);
            return va != vb ? va > vb : a < b;
        });
        for (std::int32_t k = 0; k < top_k; ++k) {
            expected_ids[static_cast<std::size_t>(t) * top_k + k] = ids[k];
            expected_values[static_cast<std::size_t>(t) * top_k + k] =
                bf16_to_f32(input[static_cast<std::size_t>(t) * vocab + ids[k]]);
        }
    }

    GuardedDeviceBuffer device_logits(input.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_ids(expected_ids.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_values(expected_values.size() * sizeof(float));
    device_logits.copy_from_host(input.data(), device_logits.bytes());
    device_ids.fill(0xcd);
    device_values.fill(0xcd);

    Tensor logits_tensor(device_logits.data(), DType::BF16, {vocab, tokens});
    Tensor ids_tensor(device_ids.data(), DType::I32, {top_k, tokens});
    Tensor values_tensor(device_values.data(), DType::FP32, {top_k, tokens});
    ops::dflash2_select_candidates(logits_tensor, ids_tensor, values_tensor, nullptr);
    cuda_synchronize();

    const std::string label = "dflash2_select_candidates";
    int failures = verify_exact(label.c_str(),
                                from_device<std::int32_t>(device_ids.data(), expected_ids.size()),
                                expected_ids);
    failures += verify_exact((label + " values").c_str(),
                             from_device<float>(device_values.data(), expected_values.size()),
                             expected_values);
    failures += verify_exact((label + " logits preserved").c_str(),
                             from_device<std::uint16_t>(device_logits.data(), input.size()), input);
    failures += device_logits.verify_guards(label + " logits");
    failures += device_ids.verify_guards(label + " ids");
    failures += device_values.verify_guards(label + " values");
    return failures;
}

int run_selector_lattice_case() {
    constexpr std::int32_t rank         = kDFlash2SelectorRank;
    constexpr std::int32_t top_k        = kDFlash2SelectorTopK;
    constexpr std::int32_t block_tokens = 4;
    constexpr std::int32_t tokens       = block_tokens;
    constexpr std::int32_t packed_width = top_k + top_k * top_k + 28;

    std::vector<float> hidden(static_cast<std::size_t>(rank) * tokens, 1.0f);
    std::vector<float> successor(static_cast<std::size_t>(rank) * top_k * tokens);
    std::vector<float> predecessor(successor.size(), 1.0f);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t s = 0; s < top_k; ++s) {
            for (std::int32_t r = 0; r < rank; ++r) {
                successor[(static_cast<std::size_t>(t) * top_k + s) * rank + r] =
                    static_cast<float>(s + 1);
            }
        }
    }
    round_to_bf16(hidden);
    round_to_bf16(successor);
    round_to_bf16(predecessor);
    const std::vector<std::uint16_t> hidden_bits       = bf16_bits(hidden);
    const std::vector<std::uint16_t> successor_bits   = bf16_bits(successor);
    const std::vector<std::uint16_t> predecessor_bits = bf16_bits(predecessor);

    std::vector<std::int32_t> candidates(static_cast<std::size_t>(top_k) * tokens);
    std::vector<float> unary(candidates.size());
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t s = 0; s < top_k; ++s) {
            candidates[static_cast<std::size_t>(t) * top_k + s] = 5000 + 100 * t + s;
            unary[static_cast<std::size_t>(t) * top_k + s] = 0.25f * static_cast<float>(s);
        }
    }

    std::vector<float> expected(static_cast<std::size_t>(packed_width) * tokens, 0.0f);
    for (std::int32_t t = 1; t < tokens; ++t) {
        const std::size_t row = static_cast<std::size_t>(t) * packed_width;
        for (std::int32_t s = 0; s < top_k; ++s) {
            expected[row + s] = static_cast<float>(candidates[static_cast<std::size_t>(t) * top_k + s]);
        }
        for (std::int32_t p = 0; p < top_k; ++p) {
            for (std::int32_t s = 0; s < top_k; ++s) {
                expected[row + top_k + p * top_k + s] =
                    256.0f * static_cast<float>(s + 1) +
                    unary[static_cast<std::size_t>(t) * top_k + s];
            }
        }
    }

    GuardedDeviceBuffer device_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_successor(successor_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_predecessor(predecessor_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_candidates(candidates.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_unary(unary.size() * sizeof(float));
    GuardedDeviceBuffer device_out(expected.size() * sizeof(float));
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_successor.copy_from_host(successor_bits.data(), device_successor.bytes());
    device_predecessor.copy_from_host(predecessor_bits.data(), device_predecessor.bytes());
    device_candidates.copy_from_host(candidates.data(), device_candidates.bytes());
    device_unary.copy_from_host(unary.data(), device_unary.bytes());
    device_out.fill(0xcd);

    Tensor hidden_tensor(device_hidden.data(), DType::BF16, {rank, tokens});
    Tensor successor_tensor(device_successor.data(), DType::BF16, {rank, top_k, tokens});
    Tensor predecessor_tensor(device_predecessor.data(), DType::BF16, {rank, top_k, tokens});
    Tensor candidates_tensor(device_candidates.data(), DType::I32, {top_k, tokens});
    Tensor unary_tensor(device_unary.data(), DType::FP32, {top_k, tokens});
    Tensor out_tensor(device_out.data(), DType::FP32, {packed_width, tokens});
    ops::dflash2_selector_lattice(hidden_tensor, successor_tensor, predecessor_tensor,
                                  candidates_tensor, unary_tensor, packed_width, block_tokens,
                                  out_tensor, nullptr);
    cuda_synchronize();

    const std::string label = "dflash2_selector_lattice";
    int failures = verify_exact(label.c_str(), from_device<float>(device_out.data(), expected.size()),
                                expected);
    failures += verify_exact((label + " candidates preserved").c_str(),
                             from_device<std::int32_t>(device_candidates.data(), candidates.size()),
                             candidates);
    failures += device_hidden.verify_guards(label + " hidden");
    failures += device_successor.verify_guards(label + " successor");
    failures += device_predecessor.verify_guards(label + " predecessor");
    failures += device_candidates.verify_guards(label + " candidates");
    failures += device_unary.verify_guards(label + " unary");
    failures += device_out.verify_guards(label + " output");
    return failures;
}

int run_trace_path_case() {
    constexpr std::int32_t top_k        = kDFlash2SelectorTopK;
    constexpr std::int32_t block_tokens = 4;
    constexpr std::int32_t batch        = 2;
    constexpr std::int32_t tokens       = block_tokens * batch;
    constexpr std::int32_t packed_width = top_k + top_k * top_k + 28;

    std::vector<float> lattice(static_cast<std::size_t>(packed_width) * tokens, -1000.0f);
    const std::int32_t chosen[batch][block_tokens - 1] = {{5, 2, 9}, {3, 14, 1}};
    const std::int32_t predecessors[batch][block_tokens - 1] = {{0, 5, 2}, {0, 3, 14}};
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t pos = 0; pos < block_tokens; ++pos) {
            const std::int32_t t = b * block_tokens + pos;
            const std::size_t row = static_cast<std::size_t>(t) * packed_width;
            for (std::int32_t s = 0; s < top_k; ++s) {
                lattice[row + s] = static_cast<float>(10000 + 100 * b + 10 * pos + s);
            }
            if (pos == 0) {
                continue;
            }
            const std::int32_t predecessor = predecessors[b][pos - 1];
            const std::int32_t next       = chosen[b][pos - 1];
            const std::size_t score_row = row + top_k + predecessor * top_k;
            lattice[score_row + next] = 100.0f + static_cast<float>(pos);
            // A tie at an unrelated predecessor must not perturb the dependent path.
            lattice[row + top_k + ((predecessor + 1) % top_k) * top_k + ((next + 1) % top_k)] =
                100.0f + static_cast<float>(pos);
        }
    }

    std::vector<std::int32_t> expected(static_cast<std::size_t>(block_tokens - 1) * batch);
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t pos = 0; pos < block_tokens - 1; ++pos) {
            const std::int32_t t = b * block_tokens + pos + 1;
            expected[static_cast<std::size_t>(b) * (block_tokens - 1) + pos] =
                static_cast<std::int32_t>(lattice[static_cast<std::size_t>(t) * packed_width +
                                                  chosen[b][pos]]);
        }
    }

    GuardedDeviceBuffer device_lattice(lattice.size() * sizeof(float));
    GuardedDeviceBuffer device_out(expected.size() * sizeof(std::int32_t));
    device_lattice.copy_from_host(lattice.data(), device_lattice.bytes());
    device_out.fill(0xcd);

    Tensor lattice_tensor(device_lattice.data(), DType::FP32, {packed_width, tokens});
    Tensor out_tensor(device_out.data(), DType::I32, {block_tokens - 1, batch});
    ops::dflash2_trace_path(lattice_tensor, block_tokens, out_tensor, nullptr);
    cuda_synchronize();

    const std::string label = "dflash2_trace_path";
    int failures = verify_exact(label.c_str(),
                                from_device<std::int32_t>(device_out.data(), expected.size()),
                                expected);
    failures += verify_exact((label + " lattice preserved").c_str(),
                             from_device<float>(device_lattice.data(), lattice.size()), lattice);
    failures += device_lattice.verify_guards(label + " lattice");
    failures += device_out.verify_guards(label + " output");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_dynamic_conv_case(0);
    failures += run_dynamic_conv_case(1);
    failures += run_qkv_projection_case();
    failures += run_predecessor_ids_case();
    failures += run_select_candidates_case();
    failures += run_selector_lattice_case();
    failures += run_trace_path_case();
    if (std::getenv("NINFER_DFLASH2_BENCH") != nullptr) {
        failures += run_dynamic_conv_benchmark();
        failures += run_candidate_selection_benchmark();
    }
    std::cout << (failures ? "FAIL" : "OK") << " dflash2\n";
    return failures ? 1 : 0;
}
