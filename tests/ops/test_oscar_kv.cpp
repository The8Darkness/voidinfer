#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/sliding_window_attention.h"

#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kD       = 128;
constexpr int kQHeads  = 32;
constexpr int kKVHeads = 8;
constexpr int kWindow  = 64;
constexpr int kContext = 24;
constexpr int kTokens  = 2;
constexpr float kScale = 0.08838834764831844055f;
constexpr ops::AttentionHeadGeometry kGeometry{kD, kQHeads, kKVHeads};

using Head = std::array<float, kD>;

std::size_t q_index(int d, int head, int token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kQHeads) * static_cast<std::size_t>(token));
}

std::size_t kv_index(int d, int head, int token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(token));
}

std::size_t context_index(int d, int head, int slot) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(slot) +
                static_cast<std::size_t>(kWindow) * static_cast<std::size_t>(head));
}

void oscar_rotation(Head& values) {
    for (int stride = 1; stride <= 16; stride <<= 1) {
        const Head previous = values;
        for (int column = 0; column < 4; ++column) {
            for (int lane = 0; lane < 32; ++lane) {
                const int d     = lane + column * 32;
                const int peer  = (lane ^ stride) + column * 32;
                values[d]       = (lane & stride) == 0 ? previous[d] + previous[peer]
                                                        : previous[peer] - previous[d];
            }
        }
    }
    for (int span = 1; span < 4; span <<= 1) {
        const Head previous = values;
        for (int base = 0; base < 4; base += 2 * span) {
            for (int offset = 0; offset < span; ++offset) {
                const int low  = base + offset;
                const int high = base + offset + span;
                for (int lane = 0; lane < 32; ++lane) {
                    values[lane + low * 32]  = previous[lane + low * 32] +
                                               previous[lane + high * 32];
                    values[lane + high * 32] = previous[lane + low * 32] -
                                               previous[lane + high * 32];
                }
            }
        }
    }
    for (float& value : values) value *= 0.08838834764831844f;
}

struct QuantParams {
    float scale;
    float zero;
};

QuantParams oscar_quant_params(const Head& values, int bits, bool is_value) {
    const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    const float ratio = bits == 2 ? (is_value ? 0.91f : 0.93f)
                                  : bits == 3 ? (is_value ? 0.95f : 0.965f)
                                               : (is_value ? 0.98f : 0.985f);
    const float center           = (*min_it + *max_it) * 0.5f;
    const float half_span        = (*max_it - *min_it) * 0.5f * ratio;
    const float zero             = center - half_span;
    const int levels             = (1 << bits) - 1;
    const float scale             = half_span * 2.0f > 1.0e-8f
                                        ? (half_span * 2.0f) / static_cast<float>(levels)
                                        : 1.0f;
    return {.scale = scale, .zero = zero};
}

std::uint8_t oscar_code(float value, QuantParams params, int bits) {
    const float normalized = (value - params.zero) / params.scale;
    const int code = std::max(0, std::min((1 << bits) - 1, static_cast<int>(std::nearbyint(normalized))));
    return static_cast<std::uint8_t>(code);
}

std::vector<std::uint8_t> pack_oscar(const Head& values, int bits, bool is_value,
                                     QuantParams& params) {
    params = oscar_quant_params(values, bits, is_value);
    std::array<std::uint8_t, kD> codes{};
    for (int d = 0; d < kD; ++d) codes[static_cast<std::size_t>(d)] = oscar_code(values[d], params, bits);
    const int code_extent = (kD * bits + 7) / 8;
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(code_extent), 0);
    for (int byte = 0; byte < code_extent; ++byte) {
        const int first_bit = byte * 8;
        for (int bit = 0; bit < 8; ++bit) {
            const int global_bit = first_bit + bit;
            const int dimension  = global_bit / bits;
            const int bit_offset = global_bit - dimension * bits;
            if (dimension < kD && ((codes[static_cast<std::size_t>(dimension)] >> bit_offset) & 1U) != 0U) {
                packed[static_cast<std::size_t>(byte)] |= static_cast<std::uint8_t>(1U << bit);
            }
        }
    }
    return packed;
}

std::uint8_t unpack_oscar(const std::vector<std::uint8_t>& packed, int dimension, int bits) {
    const int bit   = dimension * bits;
    const int byte  = bit >> 3;
    const int shift = bit & 7;
    std::uint32_t word = packed[static_cast<std::size_t>(byte)];
    if (shift + bits > 8) {
        word |= static_cast<std::uint32_t>(packed[static_cast<std::size_t>(byte + 1)]) << 8U;
    }
    return static_cast<std::uint8_t>((word >> shift) & ((1U << bits) - 1U));
}

Head decode_oscar(const std::vector<std::uint8_t>& packed, float scale, float zero, int bits) {
    Head rotated{};
    for (int d = 0; d < kD; ++d) {
        rotated[static_cast<std::size_t>(d)] =
            static_cast<float>(unpack_oscar(packed, d, bits)) * scale + zero;
    }
    oscar_rotation(rotated);
    return rotated;
}

void attention_oracle(const std::vector<float>& q, const std::vector<float>& query_k,
                      const std::vector<float>& query_v, const std::vector<float>& context_k,
                      const std::vector<float>& context_v, std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(kD) * kQHeads * kTokens, 0.0);
    for (int token = 0; token < kTokens; ++token) {
        naive_dense_softmax_attention(
            kGeometry, 1, kContext + kTokens, static_cast<double>(kScale),
            [&](int d, int head, int) { return static_cast<double>(q[q_index(d, head, token)]); },
            [&](int d, int head, int key) {
                return key < kContext
                           ? static_cast<double>(context_k[context_index(d, head, key)])
                           : static_cast<double>(query_k[kv_index(d, head, key - kContext)]);
            },
            [&](int d, int head, int key) {
                return key < kContext
                           ? static_cast<double>(context_v[context_index(d, head, key)])
                           : static_cast<double>(query_v[kv_index(d, head, key - kContext)]);
            },
            [](int, int) { return true; },
            [&](int d, int head, int, double value) { out[q_index(d, head, token)] = value; });
    }
}

double relative_l2(const std::vector<double>& actual, const std::vector<double>& expected) {
    double error = 0.0;
    double norm  = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double delta = actual[i] - expected[i];
        error += delta * delta;
        norm += expected[i] * expected[i];
    }
    return std::sqrt(error / std::max(norm, 1.0e-24));
}

std::vector<double> rotate_output(const std::vector<double>& input) {
    std::vector<double> output(input.size());
    Head head{};
    for (int token = 0; token < kTokens; ++token) {
        for (int q_head = 0; q_head < kQHeads; ++q_head) {
            for (int d = 0; d < kD; ++d) {
                head[static_cast<std::size_t>(d)] =
                    static_cast<float>(input[q_index(d, q_head, token)]);
            }
            oscar_rotation(head);
            for (int d = 0; d < kD; ++d) {
                output[q_index(d, q_head, token)] =
                    static_cast<double>(head[static_cast<std::size_t>(d)]);
            }
        }
    }
    return output;
}

int run_case(int bits, bool protect) {
    const int protected_capacity        = protect ? 8 : 0;
    const int protected_anchor_capacity = protect ? 2 : 0;
    const int protected_padded_capacity = protected_capacity + protected_anchor_capacity;
    const int code_extent               = (kD * bits + 7) / 8;
    const std::size_t input_count = static_cast<std::size_t>(kD) * kKVHeads * kContext;
    const std::size_t q_count     = static_cast<std::size_t>(kD) * kQHeads * kTokens;
    const std::size_t code_count  = static_cast<std::size_t>(code_extent) * kWindow * kKVHeads;
    const std::size_t scale_count = static_cast<std::size_t>(2) * kWindow * kKVHeads;
    const std::size_t protected_count =
        static_cast<std::size_t>(kD) * protected_padded_capacity * kKVHeads;

    std::vector<float> context_k(input_count);
    std::vector<float> context_v(input_count);
    std::vector<float> q(q_count);
    std::vector<float> query_k(static_cast<std::size_t>(kD) * kKVHeads * kTokens);
    std::vector<float> query_v(static_cast<std::size_t>(kD) * kKVHeads * kTokens);
    fill_uniform(context_k, 1001U + static_cast<unsigned>(bits * 17 + protect), -0.7f, 0.7f);
    fill_uniform(context_v, 2003U + static_cast<unsigned>(bits * 19 + protect), -0.9f, 0.9f);
    fill_uniform(q, 3001U + static_cast<unsigned>(bits * 23 + protect), -0.25f, 0.25f);
    fill_uniform(query_k, 4001U + static_cast<unsigned>(bits * 29 + protect), -0.7f, 0.7f);
    fill_uniform(query_v, 5003U + static_cast<unsigned>(bits * 31 + protect), -0.9f, 0.9f);
    round_to_bf16(context_k);
    round_to_bf16(context_v);
    round_to_bf16(q);
    round_to_bf16(query_k);
    round_to_bf16(query_v);

    DeviceBuffer d_context_k = to_device_bf16(context_k);
    DeviceBuffer d_context_v = to_device_bf16(context_v);
    DeviceBuffer d_q         = to_device_bf16(q);
    DeviceBuffer d_query_k   = to_device_bf16(query_k);
    DeviceBuffer d_query_v   = to_device_bf16(query_v);
    DeviceBuffer d_positions = to_device_i32([&] {
        std::vector<int> positions(kContext);
        for (int position = 0; position < kContext; ++position) positions[position] = position;
        return positions;
    }());
    DeviceBuffer d_count = to_device<std::int32_t>({kContext});
    DeviceBuffer d_lane  = to_device<std::int32_t>({0});
    GuardedDeviceBuffer cache_k(code_count);
    GuardedDeviceBuffer cache_v(code_count);
    GuardedDeviceBuffer cache_k_scale(scale_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer cache_v_scale(scale_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer protected_k(protected_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer protected_v(protected_count * sizeof(std::uint16_t));
    cache_k.fill(0);
    cache_v.fill(0);
    cache_k_scale.fill(0);
    cache_v_scale.fill(0);
    if (protect) {
        protected_k.fill(0);
        protected_v.fill(0);
    }

    CyclicKVCacheLayerView cache{
        .k = Tensor(cache_k.data(), DType::U8, {code_extent, kWindow, kKVHeads, 1}),
        .v = Tensor(cache_v.data(), DType::U8, {code_extent, kWindow, kKVHeads, 1}),
        .capacity = kWindow,
        .padded_capacity = kWindow,
        .num_kv_heads = kKVHeads,
        .head_dim = kD,
        .lane_capacity = 1,
        .k_scale = Tensor(cache_k_scale.data(), DType::BF16, {2, kWindow, kKVHeads, 1}),
        .v_scale = Tensor(cache_v_scale.data(), DType::BF16, {2, kWindow, kKVHeads, 1}),
        .protected_k = protect
                             ? Tensor(protected_k.data(), DType::BF16,
                                      {kD, protected_padded_capacity, kKVHeads, 1})
                             : Tensor{},
        .protected_v = protect
                             ? Tensor(protected_v.data(), DType::BF16,
                                      {kD, protected_padded_capacity, kKVHeads, 1})
                             : Tensor{},
        .protected_capacity = static_cast<std::uint32_t>(protected_capacity),
        .protected_anchor_capacity = static_cast<std::uint32_t>(protected_anchor_capacity),
        .protected_padded_capacity = static_cast<std::uint32_t>(protected_padded_capacity),
        .dtype = DType::U8,
        .quant_group = 128,
        .quant_bits = static_cast<std::uint8_t>(bits),
        .quantization = CyclicKVCacheQuantization::OscarAffine,
    };

    const Tensor input_k(d_context_k.p, DType::BF16, {kD, kKVHeads, kContext, 1});
    const Tensor input_v(d_context_v.p, DType::BF16, {kD, kKVHeads, kContext, 1});
    const Tensor positions(d_positions.p, DType::I32, {kContext, 1});
    const Tensor count(d_count.p, DType::I32, {1});
    const Tensor lane(d_lane.p, DType::I32, {1});
    ops::kv_cache_append_prefix(input_k, input_v, positions, count, lane, {0, kContext}, cache,
                                nullptr);
    cuda_synchronize();

    const std::vector<std::uint8_t> host_k_codes = from_device<std::uint8_t>(cache_k.data(), code_count);
    const std::vector<std::uint8_t> host_v_codes = from_device<std::uint8_t>(cache_v.data(), code_count);
    const std::vector<std::uint16_t> host_k_meta =
        from_device<std::uint16_t>(cache_k_scale.data(), scale_count);
    const std::vector<std::uint16_t> host_v_meta =
        from_device<std::uint16_t>(cache_v_scale.data(), scale_count);
    const std::vector<std::uint16_t> host_protected_k =
        protect ? from_device<std::uint16_t>(protected_k.data(), protected_count)
                : std::vector<std::uint16_t>{};
    const std::vector<std::uint16_t> host_protected_v =
        protect ? from_device<std::uint16_t>(protected_v.data(), protected_count)
                : std::vector<std::uint16_t>{};

    std::vector<std::uint8_t> expected_k_codes(code_count, 0);
    std::vector<std::uint8_t> expected_v_codes(code_count, 0);
    std::vector<std::uint16_t> expected_k_meta(scale_count, 0);
    std::vector<std::uint16_t> expected_v_meta(scale_count, 0);
    std::vector<std::uint16_t> expected_protected_k(protected_count, 0);
    std::vector<std::uint16_t> expected_protected_v(protected_count, 0);
    for (int position = 0; position < kContext; ++position) {
        const int slot = position & (kWindow - 1);
        const bool anchor = protect && position < protected_anchor_capacity;
        const int side_slot = anchor
                                  ? position
                                  : protected_anchor_capacity +
                                        (position & (protected_capacity - 1));
        for (int head = 0; head < kKVHeads; ++head) {
            Head raw_k{};
            Head raw_v{};
            for (int d = 0; d < kD; ++d) {
                raw_k[static_cast<std::size_t>(d)] = context_k[kv_index(d, head, position)];
                raw_v[static_cast<std::size_t>(d)] = context_v[kv_index(d, head, position)];
            }
            oscar_rotation(raw_k);
            oscar_rotation(raw_v);
            QuantParams k_params{};
            QuantParams v_params{};
            const auto expected_k = pack_oscar(raw_k, bits, false, k_params);
            const auto expected_v = pack_oscar(raw_v, bits, true, v_params);
            const std::size_t code_base = static_cast<std::size_t>(code_extent) *
                                          (slot + static_cast<std::size_t>(kWindow) * head);
            std::copy(expected_k.begin(), expected_k.end(), expected_k_codes.begin() + code_base);
            std::copy(expected_v.begin(), expected_v.end(), expected_v_codes.begin() + code_base);
            const std::size_t meta_base = static_cast<std::size_t>(2) *
                                          (slot + static_cast<std::size_t>(kWindow) * head);
            expected_k_meta[meta_base]     = f32_to_bf16(k_params.scale);
            expected_k_meta[meta_base + 1] = f32_to_bf16(k_params.zero);
            expected_v_meta[meta_base]     = f32_to_bf16(v_params.scale);
            expected_v_meta[meta_base + 1] = f32_to_bf16(v_params.zero);
            const bool recent = protect && position >= kContext - protected_capacity;
            if (protect && (anchor || recent)) {
                const std::size_t side_base = static_cast<std::size_t>(kD) *
                                              (side_slot + static_cast<std::size_t>(protected_padded_capacity) * head);
                for (int d = 0; d < kD; ++d) {
                    expected_protected_k[side_base + static_cast<std::size_t>(d)] =
                        f32_to_bf16(raw_k[static_cast<std::size_t>(d)]);
                    expected_protected_v[side_base + static_cast<std::size_t>(d)] =
                        f32_to_bf16(raw_v[static_cast<std::size_t>(d)]);
                }
            }
        }
    }
    int failures = 0;
    failures += verify_exact("OSCAR append K bytes", host_k_codes, expected_k_codes);
    failures += verify_exact("OSCAR append V bytes", host_v_codes, expected_v_codes);
    failures += verify_exact("OSCAR append K metadata", host_k_meta, expected_k_meta);
    failures += verify_exact("OSCAR append V metadata", host_v_meta, expected_v_meta);
    if (protect) {
        failures += verify_exact("OSCAR append protected K", host_protected_k, expected_protected_k);
        failures += verify_exact("OSCAR append protected V", host_protected_v, expected_protected_v);
    }

    std::vector<float> reconstructed_k(static_cast<std::size_t>(kD) * kWindow * kKVHeads, 0.0f);
    std::vector<float> reconstructed_v(reconstructed_k.size(), 0.0f);
    for (int position = 0; position < kContext; ++position) {
        const bool anchor  = protect && position < protected_anchor_capacity;
        const bool recent  = protect && position >= kContext - protected_capacity;
        const int slot     = position & (kWindow - 1);
        const int side_slot = anchor ? position
                                     : protected_anchor_capacity +
                                           (position & (protected_capacity - 1));
        for (int head = 0; head < kKVHeads; ++head) {
            Head decoded_k{};
            Head decoded_v{};
            if (anchor || recent) {
                for (int d = 0; d < kD; ++d) {
                    const std::size_t source = static_cast<std::size_t>(d) +
                        static_cast<std::size_t>(kD) *
                            (side_slot + static_cast<std::size_t>(protected_padded_capacity) * head);
                    decoded_k[static_cast<std::size_t>(d)] = bf16_to_f32(host_protected_k[source]);
                    decoded_v[static_cast<std::size_t>(d)] = bf16_to_f32(host_protected_v[source]);
                }
                oscar_rotation(decoded_k);
                oscar_rotation(decoded_v);
            } else {
                const std::size_t code_base = static_cast<std::size_t>(code_extent) *
                                              (slot + static_cast<std::size_t>(kWindow) * head);
                const std::size_t scale_base = static_cast<std::size_t>(2) *
                                               (slot + static_cast<std::size_t>(kWindow) * head);
                std::vector<std::uint8_t> packed_k(host_k_codes.begin() + code_base,
                                                   host_k_codes.begin() + code_base + code_extent);
                std::vector<std::uint8_t> packed_v(host_v_codes.begin() + code_base,
                                                   host_v_codes.begin() + code_base + code_extent);
                decoded_k = decode_oscar(packed_k, bf16_to_f32(host_k_meta[scale_base]),
                                         bf16_to_f32(host_k_meta[scale_base + 1]), bits);
                decoded_v = decode_oscar(packed_v, bf16_to_f32(host_v_meta[scale_base]),
                                         bf16_to_f32(host_v_meta[scale_base + 1]), bits);
            }
            for (int d = 0; d < kD; ++d) {
                reconstructed_k[context_index(d, head, slot)] = decoded_k[static_cast<std::size_t>(d)];
                reconstructed_v[context_index(d, head, slot)] = decoded_v[static_cast<std::size_t>(d)];
            }
        }
    }

    std::vector<double> reference;
    attention_oracle(q, query_k, query_v, reconstructed_k, reconstructed_v, reference);
    std::vector<float> original_k(reconstructed_k.size(), 0.0f);
    std::vector<float> original_v(reconstructed_v.size(), 0.0f);
    for (int position = 0; position < kContext; ++position) {
        for (int head = 0; head < kKVHeads; ++head) {
            for (int d = 0; d < kD; ++d) {
                original_k[context_index(d, head, position)] = context_k[kv_index(d, head, position)];
                original_v[context_index(d, head, position)] = context_v[kv_index(d, head, position)];
            }
        }
    }
    std::vector<double> fp16_reference;
    attention_oracle(q, query_k, query_v, original_k, original_v, fp16_reference);

    DeviceBuffer d_attention_positions = to_device_i32({kContext, kContext + 1});
    DeviceBuffer d_valid               = to_device<std::int32_t>({kTokens});
    GuardedDeviceBuffer d_out(q_count * sizeof(std::uint16_t));
    d_out.fill(0x7f);
    const Tensor q_tensor(d_q.p, DType::BF16, {kD, kQHeads, kTokens, 1});
    const Tensor query_k_tensor(d_query_k.p, DType::BF16, {kD, kKVHeads, kTokens, 1});
    const Tensor query_v_tensor(d_query_v.p, DType::BF16, {kD, kKVHeads, kTokens, 1});
    const Tensor attention_positions(d_attention_positions.p, DType::I32, {kTokens, 1});
    const Tensor valid(d_valid.p, DType::I32, {1});
    const Tensor attention_lane(d_lane.p, DType::I32, {1});
    Tensor out(d_out.data(), DType::BF16, {kD, kQHeads, kTokens, 1});
    DeviceArena workspace(1);
    ops::sliding_window_attention(q_tensor, query_k_tensor, query_v_tensor, attention_positions,
                                  valid, attention_lane, kGeometry, kWindow, kScale, cache,
                                  {0, kContext + 1}, workspace, out, nullptr);
    cuda_synchronize();

    const std::string label = "OSCAR-Q" + std::to_string(bits) + (protect ? " protected" : " packed");
    const auto actual = from_device_bf16(d_out.data(), q_count);
    const auto rotated_reference = rotate_output(reference);
    std::cout << label << " rel_host=" << relative_l2(actual, reference)
              << " rel_fp16=" << relative_l2(actual, fp16_reference)
              << " rel_rotated_host=" << relative_l2(actual, rotated_reference) << '\n';
    failures += verify_reduction(label.c_str(), actual, reference,
                                 {.relative_l2 = 4.5e-3,
                                  .gross_absolute = 5e-4,
                                  .gross_relative_to_max_reference = 4.5e-3});
    failures += d_out.verify_guards(label + " output guards");
    failures += cache_k.verify_guards(label + " K guards");
    failures += cache_v.verify_guards(label + " V guards");
    failures += cache_k_scale.verify_guards(label + " K metadata guards");
    failures += cache_v_scale.verify_guards(label + " V metadata guards");
    if (protect) {
        failures += protected_k.verify_guards(label + " protected K guards");
        failures += protected_v.verify_guards(label + " protected V guards");
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }
    int failures = 0;
    for (const int bits : {2, 4}) {
        failures += run_case(bits, false);
        failures += run_case(bits, true);
    }
    if (failures != 0) {
        std::cerr << "oscar_kv failures=" << failures << '\n';
        return 1;
    }
    std::cout << "oscar_kv: PASS\n";
    return 0;
}
