// E4C1 adapter unit test: loader, cursor tiers, exact small-context decode.
#include "exl3/oscar_runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

namespace {

using ninfer::exl3::Exl3OscarContext;
using ninfer::exl3::Exl3OscarCheckpoint;
using ninfer::exl3::Exl3OscarLayerCache;
using ninfer::exl3::Exl3OscarRotations;
using ninfer::exl3::Exl3OscarTelemetry;
using ninfer::exl3::exl3_oscar_load_rotations;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

std::string env(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
}

std::uint16_t float_to_half_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    const std::uint32_t sign = (bits >> 16) & 0x8000U;
    const int exponent = static_cast<int>((bits >> 23) & 255U) - 112;
    const std::uint32_t mantissa = bits & 0x7fffffU;
    if (exponent <= 0) return static_cast<std::uint16_t>(sign);
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7bffU);
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) |
                                      (mantissa >> 13));
}

float half_bits_to_float(std::uint16_t bits) {
    std::uint32_t value = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
    const std::uint32_t exponent = (bits >> 10U) & 31U;
    const std::uint32_t fraction = bits & 1023U;
    if (exponent == 0U) {
        if (fraction != 0U) {
            std::uint32_t mantissa = fraction;
            int shift = 0;
            while ((mantissa & 1024U) == 0U) { mantissa <<= 1U; ++shift; }
            value |= static_cast<std::uint32_t>(127 - 14 - shift) << 23U;
            value |= (mantissa & 1023U) << 13U;
        }
    } else if (exponent == 31U) {
        value |= 0x7f800000U | (fraction << 13U);
    } else {
        value |= (exponent + 112U) << 23U | (fraction << 13U);
    }
    float result = 0.0F;
    std::memcpy(&result, &value, 4);
    return result;
}

float bf16_round(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    bits >>= 16U;
    bits <<= 16U;
    float rounded = 0.0F;
    std::memcpy(&rounded, &bits, 4);
    return rounded;
}

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(operation);
}

std::vector<std::byte> download_bytes(const void* device, std::size_t bytes,
                                      const char* operation) {
    std::vector<std::byte> result(bytes);
    cuda_check(cudaMemcpy(result.data(), device, bytes, cudaMemcpyDeviceToHost), operation);
    return result;
}

void require_active_cache_equal(const std::vector<std::byte>& a,
                                const std::vector<std::byte>& b, std::uint32_t context,
                                const std::string& label) {
    constexpr std::size_t token_bytes = 4ULL * 256 * sizeof(std::uint16_t);
    constexpr std::size_t prefix_bytes = 64ULL * token_bytes;
    constexpr std::size_t recent_bytes = 256ULL * token_bytes;
    const std::size_t prefix_tokens = std::min<std::uint32_t>(context, 64U);
    auto equal_range = [&](std::size_t offset, std::size_t bytes) {
        return std::equal(a.begin() + offset, a.begin() + offset + bytes,
                          b.begin() + offset);
    };
    require(equal_range(0, prefix_tokens * token_bytes), label + " prefix K mismatch");
    require(equal_range(prefix_bytes, prefix_tokens * token_bytes),
            label + " prefix V mismatch");
    const std::uint32_t recent_begin =
        context <= 64U ? context : std::max(64U, context > 256U ? context - 256U : 0U);
    const std::uint32_t recent_tokens = context - recent_begin;
    const std::uint32_t head = recent_begin & 255U;
    for (std::uint32_t logical = 0; logical < recent_tokens; ++logical) {
        const std::size_t physical = (head + logical) & 255U;
        const std::size_t k_offset = 2 * prefix_bytes + physical * token_bytes;
        const std::size_t v_offset = 2 * prefix_bytes + recent_bytes + physical * token_bytes;
        require(equal_range(k_offset, token_bytes), label + " recent K mismatch");
        require(equal_range(v_offset, token_bytes), label + " recent V mismatch");
    }
}

// Host exact reference for the all-prefix (<=64 tokens) case: F16 taps ->
// FP32 -> rotate -> exact attention -> R_V^T recovery. No quantization.
std::vector<float> host_prefix_decode(const std::vector<std::uint16_t>& q_f16,
                                      const std::vector<std::uint16_t>& k_f16,
                                      const std::vector<std::uint16_t>& v_f16, int rows,
                                      const std::vector<float>& rk,
                                      const std::vector<float>& rv) {
    constexpr int H = 24, KV = 4, D = 256;
    std::vector<float> q(H * D), keys(static_cast<std::size_t>(rows) * KV * D),
        vals(static_cast<std::size_t>(rows) * KV * D);
    for (int h = 0; h < H; ++h) {
        for (int d = 0; d < D; ++d) {
            float s = 0.0F;
            for (int i = 0; i < D; ++i) {
                const float x = half_bits_to_float(q_f16[(static_cast<std::size_t>(rows - 1) * H + h) * D + i]);
                s = std::fma(x, rk[static_cast<std::size_t>(i) * D + d], s);
            }
            q[static_cast<std::size_t>(h) * D + d] = s;
        }
    }
    for (int t = 0; t < rows; ++t) {
        for (int h = 0; h < KV; ++h) {
            for (int d = 0; d < D; ++d) {
                float sk = 0.0F, sv = 0.0F;
                for (int i = 0; i < D; ++i) {
                    const float xk = half_bits_to_float(k_f16[(static_cast<std::size_t>(t) * KV + h) * D + i]);
                    const float xv = half_bits_to_float(v_f16[(static_cast<std::size_t>(t) * KV + h) * D + i]);
                    sk = std::fma(xk, rk[static_cast<std::size_t>(i) * D + d], sk);
                    sv = std::fma(xv, rv[static_cast<std::size_t>(i) * D + d], sv);
                }
                // Resident prefix rows are BF16-rounded by the write kernel.
                keys[(static_cast<std::size_t>(t) * KV + h) * D + d] = bf16_round(sk);
                vals[(static_cast<std::size_t>(t) * KV + h) * D + d] = bf16_round(sv);
            }
        }
    }
    std::vector<float> out(H * D, 0.0F);
    for (int qh = 0; qh < H; ++qh) {
        const int kvh = qh / 6;
        std::vector<float> scores(rows);
        for (int c = 0; c < rows; ++c) {
            float s = 0.0F;
            for (int d = 0; d < D; ++d) {
                s = std::fma(q[static_cast<std::size_t>(qh) * D + d],
                             keys[(static_cast<std::size_t>(c) * KV + kvh) * D + d], s);
            }
            scores[c] = s / 16.0F;
        }
        const float m = *std::max_element(scores.begin(), scores.end());
        float den = 0.0F;
        for (float& s : scores) { s = std::exp(s - m); den += s; }
        std::vector<float> av(D, 0.0F);
        for (int c = 0; c < rows; ++c) {
            const float w = scores[c] / den;
            for (int d = 0; d < D; ++d) {
                av[d] = std::fma(w, vals[(static_cast<std::size_t>(c) * KV + kvh) * D + d], av[d]);
            }
        }
        for (int d = 0; d < D; ++d) {
            float s = 0.0F;
            for (int i = 0; i < D; ++i) {
                s = std::fma(av[i], rv[static_cast<std::size_t>(d) * D + i], s);
            }
            out[static_cast<std::size_t>(qh) * D + d] = s;
        }
    }
    return out;
}

int run() {
    const std::string asset_dir = env("NINFER_OSCAR_ROTATION_ASSET_DIR");
    if (asset_dir.empty()) {
        std::cout << "E4C1 adapter test skipped: set NINFER_OSCAR_ROTATION_ASSET_DIR\n";
        return 77;
    }
    const Exl3OscarRotations rotations = exl3_oscar_load_rotations(
        asset_dir, "results/oscar/E4C1_EXL3_COMPATIBILITY_MANIFEST.json");
    require(rotations.asset_identity ==
                "qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1",
            "adapter loader identity mismatch");
    for (int b = 0; b < 16; ++b) {
        require(rotations.r_k[b].size() == 65536, "adapter RK bank size");
        require(rotations.r_v[b].size() == 65536, "adapter RV bank size");
    }
    std::cout << "adapter loader: PASS banks=16\n";

    Exl3OscarTelemetry telemetry{};
    auto oscar = Exl3OscarContext::create(rotations, 1024, &telemetry);
    require(!Exl3OscarContext::is_full_attention_layer(0), "layer 0 must not be full attention");
    require(Exl3OscarContext::is_full_attention_layer(3), "layer 3 must be full attention");
    require(Exl3OscarContext::bank_index(63) == 15, "layer 63 bank must be 15");

    // Deterministic F16 taps: per-(row,head,dim) hashed values in [-1, 1].
    constexpr int kRows = 600;
    std::vector<std::uint16_t> q_taps(static_cast<std::size_t>(kRows) * 24 * 256);
    std::vector<std::uint16_t> k_taps(static_cast<std::size_t>(kRows) * 4 * 256);
    std::vector<std::uint16_t> v_taps(static_cast<std::size_t>(kRows) * 4 * 256);
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    auto next_value = [&] {
        state = 6364136223846793005ULL * state + 1442695040888963407ULL;
        return static_cast<float>((state >> 33) & 1023U) / 512.0F - 1.0F;
    };
    for (auto& bits : q_taps) bits = float_to_half_bits(next_value());
    for (auto& bits : k_taps) bits = float_to_half_bits(next_value());
    for (auto& bits : v_taps) bits = float_to_half_bits(next_value());

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "adapter test stream");
    // Adapter taps are device buffers (as in EXL3 production); upload fixtures.
    std::uint16_t* d_qtaps = nullptr;
    std::uint16_t* d_ktaps = nullptr;
    std::uint16_t* d_vtaps = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_qtaps), q_taps.size() * 2), "taps q alloc");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ktaps), k_taps.size() * 2), "taps k alloc");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_vtaps), v_taps.size() * 2), "taps v alloc");
    cuda_check(cudaMemcpy(d_qtaps, q_taps.data(), q_taps.size() * 2, cudaMemcpyHostToDevice),
               "taps q upload");
    cuda_check(cudaMemcpy(d_ktaps, k_taps.data(), k_taps.size() * 2, cudaMemcpyHostToDevice),
               "taps k upload");
    cuda_check(cudaMemcpy(d_vtaps, v_taps.data(), v_taps.size() * 2, cudaMemcpyHostToDevice),
               "taps v upload");
    // Chunked appends crossing 64 / 320 / 512 tier boundaries.
    std::uint32_t start = 0;
    for (int chunk : {1, 13, 50, 1, 255, 1, 191, 88}) {
        oscar->append_layer(3, d_qtaps + static_cast<std::size_t>(start) * 24 * 256,
                            d_ktaps + static_cast<std::size_t>(start) * 4 * 256,
                            d_vtaps + static_cast<std::size_t>(start) * 4 * 256, chunk,
                            start, stream);
        start += static_cast<std::uint32_t>(chunk);
    }
    require(start == 600, "adapter append accounting");
    cuda_check(cudaStreamSynchronize(stream), "adapter append sync");
    cuda_check(cudaGetLastError(), "adapter append kernel error");

    // Exact small-context check at 64 rows on a fresh context (all prefix,
    // 4 tokens per S16 split, no empty split ranges).
    oscar->reset();
    oscar->append_layer(3, d_qtaps, d_ktaps, d_vtaps, 64, 0, stream);
    std::vector<std::uint16_t> attn(24 * 256, 0);
    std::vector<std::uint16_t> q_row(24 * 256);
    std::copy(q_taps.begin() + static_cast<std::size_t>(63) * 24 * 256,
              q_taps.begin() + static_cast<std::size_t>(64) * 24 * 256, q_row.begin());
    // Upload the single query row.
    std::uint16_t* d_q = nullptr;
    std::uint16_t* d_out = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_q), q_row.size() * 2), "adapter test q alloc");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_out), attn.size() * 2), "adapter test out alloc");
    cuda_check(cudaMemcpy(d_q, q_row.data(), q_row.size() * 2, cudaMemcpyHostToDevice),
               "adapter test q upload");
    cuda_check(cudaGetLastError(), "adapter pre-decode error");
    oscar->decode_layer(3, d_q, d_out, 63, 0, stream);
    cuda_check(cudaStreamSynchronize(stream), "adapter decode sync");
    cuda_check(cudaGetLastError(), "adapter decode kernel error");
    cuda_check(cudaMemcpy(attn.data(), d_out, attn.size() * 2, cudaMemcpyDeviceToHost),
               "adapter test out download");
    const std::vector<float> expected =
        host_prefix_decode(q_taps, k_taps, v_taps, 64, rotations.r_k[0], rotations.r_v[0]);
    double error_sq = 0.0, expected_sq = 0.0;
    double max_abs = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double actual = half_bits_to_float(attn[i]);
        require(std::isfinite(actual), "adapter decode produced non-finite output");
        const double error = actual - expected[i];
        max_abs = std::max(max_abs, std::abs(error));
        error_sq += error * error;
        expected_sq += expected[i] * expected[i];
    }
    const double rel_l2 = std::sqrt(error_sq) / std::max(std::sqrt(expected_sq), 1.0e-30);
    std::cout << "adapter exact prefix decode: max_abs=" << max_abs << " rel_l2=" << rel_l2 << "\n";
    require(max_abs <= 0.01 && rel_l2 <= 5e-4, "adapter exact prefix decode diverged");

    // Determinism: same query twice -> identical bits.
    std::vector<std::uint16_t> attn2(24 * 256, 0);
    oscar->decode_layer(3, d_q, d_out, 63, 0, stream);
    cuda_check(cudaStreamSynchronize(stream), "adapter decode2 sync");
    cuda_check(cudaMemcpy(attn2.data(), d_out, attn2.size() * 2, cudaMemcpyDeviceToHost),
               "adapter test out2 download");
    require(attn == attn2, "adapter decode is not deterministic");
    require(telemetry.last_split_class == 16, "context 64 must select S16");
    std::cout << "adapter determinism + split class: PASS split=" << telemetry.last_split_class
              << "\n";

    // Eager transaction boundaries: compare one accepted continuation against
    // an uninterrupted context after discarding and rolling back a B8 suffix.
    Exl3OscarTelemetry reference_telemetry{};
    auto reference = Exl3OscarContext::create(rotations, 1024, &reference_telemetry);
    oscar->reset();
    const std::size_t checkpoint_bytes = oscar->checkpoint_device_bytes();
    require(checkpoint_bytes == 20ULL * 1024 * 1024,
            "OSCAR context checkpoint must be exactly 20 MiB");
    void* d_checkpoint_a = nullptr;
    void* d_checkpoint_b = nullptr;
    cuda_check(cudaMalloc(&d_checkpoint_a, checkpoint_bytes), "adapter checkpoint A alloc");
    cuda_check(cudaMalloc(&d_checkpoint_b, checkpoint_bytes), "adapter checkpoint B alloc");
    Exl3OscarCheckpoint checkpoint_a{d_checkpoint_a, checkpoint_bytes};
    Exl3OscarCheckpoint checkpoint_b{d_checkpoint_b, checkpoint_bytes};
    bool graph_save_rejected = false;
    oscar->set_graph_class(16);
    try { oscar->save_checkpoint(checkpoint_a, stream); }
    catch (const std::logic_error&) { graph_save_rejected = true; }
    oscar->set_graph_class(0);
    require(graph_save_rejected, "OSCAR graph-mode checkpoint save was not rejected");
    oscar->save_checkpoint(checkpoint_a, stream);
    oscar->reset();
    bool reset_snapshot_rejected = false;
    try { oscar->restore_checkpoint(checkpoint_a, stream); }
    catch (const std::invalid_argument&) { reset_snapshot_rejected = true; }
    require(reset_snapshot_rejected, "OSCAR reset did not invalidate its checkpoint");
    oscar->save_checkpoint(checkpoint_a, stream);
    oscar->save_checkpoint(checkpoint_b, stream);
    bool superseded_snapshot_rejected = false;
    try { oscar->restore_checkpoint(checkpoint_a, stream); }
    catch (const std::invalid_argument&) { superseded_snapshot_rejected = true; }
    require(superseded_snapshot_rejected,
            "OSCAR newer save did not invalidate the older checkpoint");

    std::uint16_t* d_reference_out = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_reference_out), attn.size() * 2),
               "adapter reference out alloc");
    auto append_rows = [&](Exl3OscarContext& context, std::uint32_t first, int rows) {
        context.append_layer(3,
            d_qtaps + static_cast<std::size_t>(first) * 24 * 256,
            d_ktaps + static_cast<std::size_t>(first) * 4 * 256,
            d_vtaps + static_cast<std::size_t>(first) * 4 * 256,
            rows, first, stream);
    };
    auto append_decode = [&](Exl3OscarContext& context, std::uint32_t row,
                             std::uint16_t* destination) {
        append_rows(context, row, 1);
        context.decode_layer(3, d_qtaps + static_cast<std::size_t>(row) * 24 * 256,
                             destination, row, 0, stream);
    };
    std::uint32_t current = 0;
    bool graph_restore_rejected = false;
    for (std::uint32_t boundary : {63U, 64U, 65U, 319U, 320U, 321U, 575U, 576U}) {
        if (boundary > current) {
            const int gap = static_cast<int>(boundary - current);
            append_rows(*oscar, current, gap);
            append_rows(*reference, current, gap);
            current = boundary;
        }
        oscar->save_checkpoint(checkpoint_a, stream);
        const auto saved_contexts = checkpoint_a.contexts;
        const auto saved_heads = checkpoint_a.recent_heads;
        const auto saved_generation = checkpoint_a.generation;
        if (boundary == 63U) {
            oscar->set_graph_class(16);
            try { oscar->restore_checkpoint(checkpoint_a, stream); }
            catch (const std::logic_error&) { graph_restore_rejected = true; }
            oscar->set_graph_class(0);
        }
        for (std::uint32_t discarded = 0; discarded < 8; ++discarded) {
            append_decode(*oscar, boundary + discarded, d_out);
        }
        oscar->restore_checkpoint(checkpoint_a, stream);
        oscar->restore_checkpoint(checkpoint_a, stream);
        oscar->save_checkpoint(checkpoint_b, stream);
        cuda_check(cudaStreamSynchronize(stream), "adapter restored checkpoint sync");
        require(checkpoint_b.contexts == checkpoint_a.contexts &&
                    checkpoint_b.recent_heads == checkpoint_a.recent_heads,
                "OSCAR restored host extents mismatch");
        require(download_bytes(d_checkpoint_a, checkpoint_bytes,
                               "adapter checkpoint A download") ==
                    download_bytes(d_checkpoint_b, checkpoint_bytes,
                                   "adapter checkpoint B download"),
                "OSCAR restored prefix/recent bytes mismatch");
        require(checkpoint_a.contexts == saved_contexts &&
                    checkpoint_a.recent_heads == saved_heads &&
                    checkpoint_a.generation == saved_generation,
                "OSCAR restore modified caller checkpoint metadata");
        append_decode(*oscar, boundary, d_out);
        append_decode(*reference, boundary, d_reference_out);
        cuda_check(cudaStreamSynchronize(stream), "adapter continuation sync");
        require(download_bytes(d_out, attn.size() * 2, "adapter transaction output download") ==
                    download_bytes(d_reference_out, attn.size() * 2,
                                   "adapter reference output download"),
                "OSCAR continuation output mismatch at boundary " +
                    std::to_string(boundary));
        oscar->save_checkpoint(checkpoint_a, stream);
        reference->save_checkpoint(checkpoint_b, stream);
        cuda_check(cudaStreamSynchronize(stream), "adapter continuation checkpoint sync");
        const auto transaction_cache =
            download_bytes(d_checkpoint_a, checkpoint_bytes, "adapter transaction cache download");
        const auto reference_cache =
            download_bytes(d_checkpoint_b, checkpoint_bytes, "adapter reference cache download");
        require_active_cache_equal(transaction_cache, reference_cache, boundary + 1,
                                   "OSCAR continuation at " + std::to_string(boundary));
        ++current;
    }
    require(graph_restore_rejected, "OSCAR graph-mode checkpoint restore was not rejected");
    std::cout << "adapter eager transaction boundaries: PASS B8={63,64,65,319,320,321,575,576} bytes="
              << checkpoint_bytes << "\n";

    // Historical rows are append-only: speculative aging at 576 may publish a
    // suffix, but it must leave all 256 previously committed rows byte-exact.
    constexpr int history_rows = 584;
    std::vector<float> rotated_k(static_cast<std::size_t>(history_rows) * 4 * 256);
    std::vector<float> rotated_v(static_cast<std::size_t>(history_rows) * 4 * 256);
    for (std::size_t i = 0; i < rotated_k.size(); ++i) {
        rotated_k[i] = half_bits_to_float(k_taps[i]);
        rotated_v[i] = half_bits_to_float(v_taps[i]);
    }
    float* d_rotated_k = nullptr;
    float* d_rotated_v = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rotated_k), rotated_k.size() * 4),
               "adapter history K alloc");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rotated_v), rotated_v.size() * 4),
               "adapter history V alloc");
    cuda_check(cudaMemcpy(d_rotated_k, rotated_k.data(), rotated_k.size() * 4,
                          cudaMemcpyHostToDevice), "adapter history K upload");
    cuda_check(cudaMemcpy(d_rotated_v, rotated_v.data(), rotated_v.size() * 4,
                          cudaMemcpyHostToDevice), "adapter history V upload");
    Exl3OscarLayerCache history_cache(3, 0, 1024);
    // Match Context's bounded append chunks so every claimed historical row
    // is initialized through the real recent-to-history aging path.
    history_cache.append(d_rotated_k, d_rotated_v, 256, 0, stream);
    history_cache.append(d_rotated_k + 256ULL * 4 * 256,
                         d_rotated_v + 256ULL * 4 * 256, 256, 256, stream);
    history_cache.append(d_rotated_k + 512ULL * 4 * 256,
                         d_rotated_v + 512ULL * 4 * 256, 64, 512, stream);
    cuda_check(cudaStreamSynchronize(stream), "adapter committed history sync");
    auto history_view = history_cache.view();
    constexpr std::size_t committed_payload_bytes = 256ULL * 4 * 64;
    constexpr std::size_t committed_metadata_bytes = 256ULL * 4 * 4 * sizeof(float);
    const auto committed_k = download_bytes(history_view.hist_k, committed_payload_bytes,
                                            "adapter committed history K download");
    const auto committed_v = download_bytes(history_view.hist_v, committed_payload_bytes,
                                            "adapter committed history V download");
    const auto committed_k_meta = download_bytes(history_view.hist_k_meta,
                                                 committed_metadata_bytes,
                                                 "adapter committed history K meta download");
    const auto committed_v_meta = download_bytes(history_view.hist_v_meta,
                                                 committed_metadata_bytes,
                                                 "adapter committed history V meta download");
    history_cache.append(d_rotated_k + 576ULL * 4 * 256,
                         d_rotated_v + 576ULL * 4 * 256, 8, 576, stream);
    cuda_check(cudaStreamSynchronize(stream), "adapter discarded history sync");
    require(download_bytes(history_view.hist_k, committed_payload_bytes,
                           "adapter post-discard history K download") == committed_k,
            "OSCAR discarded suffix overwrote committed history K");
    require(download_bytes(history_view.hist_v, committed_payload_bytes,
                           "adapter post-discard history V download") == committed_v,
            "OSCAR discarded suffix overwrote committed history V");
    require(download_bytes(history_view.hist_k_meta, committed_metadata_bytes,
                           "adapter post-discard history K meta download") == committed_k_meta,
            "OSCAR discarded suffix overwrote committed history K metadata");
    require(download_bytes(history_view.hist_v_meta, committed_metadata_bytes,
                           "adapter post-discard history V meta download") == committed_v_meta,
            "OSCAR discarded suffix overwrote committed history V metadata");
    std::int32_t prefix = 0, historical = 0, recent = 0, ring_head = 0;
    history_cache.prefix_extent(prefix, historical, recent, ring_head);
    require(prefix == 64 && historical == 264 && recent == 256,
            "OSCAR discarded historical suffix extent mismatch");
    std::cout << "adapter append-only committed history: PASS rows=256 discarded_suffix=8\n";

    // Default-off operator screen for the fixed-B8 device-positioned
    // chronological OSCAR cohort. This deliberately stops below model
    // integration: it compares eight true sequential decode operations with
    // one cohort at every state/split boundary that can change addressing.
    if (env("NINFER_OSCAR_CONTINUATION_COHORT_B8") == "1") {
        constexpr int kCohortRows = 8;
        // The eager route gate also probes the upper adaptive boundary at
        // 8191/8192. Keep the fixture one cohort wider than the selected
        // max-context so the last row remains addressable.
        constexpr int kCohortMaxContext = 16384;
        constexpr int kCohortFixtureRows = kCohortMaxContext + kCohortRows;
        constexpr std::size_t kQRowValues = 24ULL * 256;
        constexpr std::size_t kKVRowValues = 4ULL * 256;
        constexpr std::size_t kOutValues = kCohortRows * kQRowValues;
        const std::vector<int> bases{32,319,320,511,512,767,4096};
        std::vector<std::uint16_t> cohort_q(kOutValues);
        std::vector<std::uint16_t> cohort_k(
            static_cast<std::size_t>(kCohortFixtureRows) * kKVRowValues);
        std::vector<std::uint16_t> cohort_v(cohort_k.size());
        std::uint64_t cohort_state = 0xd1b54a32d192ed03ULL;
        const auto cohort_value = [&]() {
            cohort_state = cohort_state * 2862933555777941757ULL +
                3037000493ULL;
            return static_cast<float>((cohort_state >> 35) & 2047U) /
                1024.0F - 1.0F;
        };
        for (auto& bits : cohort_q) bits = float_to_half_bits(cohort_value());
        for (auto& bits : cohort_k) bits = float_to_half_bits(cohort_value());
        for (auto& bits : cohort_v) bits = float_to_half_bits(cohort_value());

        std::uint16_t* d_cohort_q = nullptr;
        std::uint16_t* d_cohort_k = nullptr;
        std::uint16_t* d_cohort_v = nullptr;
        std::uint16_t* d_sequential_out = nullptr;
        std::uint16_t* d_cohort_out = nullptr;
        int* d_sequential_position = nullptr;
        int* d_cohort_position = nullptr;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cohort_q),
                              cohort_q.size()*sizeof(std::uint16_t)),
                   "cohort Q alloc");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cohort_k),
                              cohort_k.size()*sizeof(std::uint16_t)),
                   "cohort K alloc");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cohort_v),
                              cohort_v.size()*sizeof(std::uint16_t)),
                   "cohort V alloc");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sequential_out),
                              kOutValues*sizeof(std::uint16_t)),
                   "cohort sequential output alloc");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cohort_out),
                              kOutValues*sizeof(std::uint16_t)),
                   "cohort output alloc");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sequential_position),sizeof(int)),
                   "cohort sequential position alloc");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cohort_position),sizeof(int)),
                   "cohort position alloc");
        cuda_check(cudaMemcpy(d_cohort_q,cohort_q.data(),
                              cohort_q.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice),
                   "cohort Q upload");
        cuda_check(cudaMemcpy(d_cohort_k,cohort_k.data(),
                              cohort_k.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice),
                   "cohort K upload");
        cuda_check(cudaMemcpy(d_cohort_v,cohort_v.data(),
                              cohort_v.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice),
                   "cohort V upload");

        Exl3OscarTelemetry sequential_telemetry{},cohort_telemetry{};
        auto sequential=Exl3OscarContext::create(
            rotations,kCohortMaxContext,&sequential_telemetry);
        auto cohort=Exl3OscarContext::create(
            rotations,kCohortMaxContext,&cohort_telemetry);
        require(cohort->continuation_cohort_b8_enabled(),
                "OSCAR cohort B8 flag was not admitted");
        // This also pre-arms the chronological dynamic-shared-memory attribute;
        // the candidate launcher itself remains capture-safe and launch-only.
        sequential->ensure_device_attributes();
        cohort->ensure_device_attributes();
        const auto split_for_total=[](int total) {
            return total<=512?16:(total<=8192?32:64);
        };
        const auto append_prefix=[&](Exl3OscarContext& context,int base) {
            context.append_kv_layer(3,d_cohort_k,d_cohort_v,base,0,stream);
            context.sync_device_state(stream);
        };
        for (const int base : bases) {
            sequential->reset();
            cohort->reset();
            append_prefix(*sequential,base);
            append_prefix(*cohort,base);
            const auto sequential_dispatch_before=sequential_telemetry.oscar_dispatches[3];
            const auto cohort_dispatch_before=cohort_telemetry.oscar_dispatches[3];
            for (int row=0;row<kCohortRows;++row) {
                const int position=base+row;
                cuda_check(cudaMemcpyAsync(d_sequential_position,&position,sizeof(position),
                                            cudaMemcpyHostToDevice,stream),
                           "cohort sequential position upload");
                sequential->set_graph_class(split_for_total(position+1));
                sequential->forward_decode_device(
                    3,d_cohort_q+static_cast<std::size_t>(row)*kQRowValues,
                    d_cohort_k+static_cast<std::size_t>(position)*kKVRowValues,
                    d_cohort_v+static_cast<std::size_t>(position)*kKVRowValues,
                    d_sequential_out+static_cast<std::size_t>(row)*kQRowValues,
                    d_sequential_position,stream);
            }
            cuda_check(cudaMemcpyAsync(d_cohort_position,&base,sizeof(base),
                                        cudaMemcpyHostToDevice,stream),
                       "cohort position upload");
            cohort->set_graph_class(split_for_total(base+kCohortRows));
            cohort->forward_continuation_device_cohort_b8(
                3,d_cohort_q,
                d_cohort_k+static_cast<std::size_t>(base)*kKVRowValues,
                d_cohort_v+static_cast<std::size_t>(base)*kKVRowValues,
                d_cohort_out,d_cohort_position,cohort->graph_class(),stream);
            cuda_check(cudaStreamSynchronize(stream),"cohort exact sync");
            require(download_bytes(d_sequential_out,kOutValues*sizeof(std::uint16_t),
                                   "cohort sequential output download") ==
                        download_bytes(d_cohort_out,kOutValues*sizeof(std::uint16_t),
                                       "cohort output download"),
                    "OSCAR cohort B8 output mismatch at base " + std::to_string(base));
            require(sequential->graph_layer_state_host_for_test(3,stream) ==
                        cohort->graph_layer_state_host_for_test(3,stream),
                    "OSCAR cohort B8 raw state mismatch at base " +
                        std::to_string(base));
            require(sequential_telemetry.oscar_dispatches[3] ==
                        sequential_dispatch_before &&
                    cohort_telemetry.oscar_dispatches[3] == cohort_dispatch_before,
                    "OSCAR cohort operator emitted phantom route telemetry");
        }

        // Direct eager-admission gate. Stable-class cohorts must be byte-exact
        // against the existing row-ordered device path and must publish only
        // the selected layer's host/telemetry state. A cohort that crosses an
        // adaptive 512/8192 split is rejected before launch so the caller can
        // retain its chronological eager fallback.
        const std::vector<int> eager_rows{2,4,8};
        const std::vector<int> eager_bases{511,512,8191,8192};
        const auto eager_attempts_before=cohort_telemetry.continuation_cohort_eager_attempts;
        const auto eager_dispatches_before=cohort_telemetry.continuation_cohort_eager_dispatches;
        const auto eager_latch_before=cohort_telemetry.continuation_cohort_eager_latch_misses;
        const auto eager_boundary_before=cohort_telemetry.continuation_cohort_eager_boundary_fallbacks;
        const auto eager_malformed_before=cohort_telemetry.continuation_cohort_eager_malformed;
        const auto eager_recent_begin=[](int context) {
            return context<=64?context:(context>256?std::max(64,context-256):64);
        };
        const auto assert_eager_live_layer=[&](int expected_context) {
            const auto live=cohort->live_state_host_for_test(stream);
            constexpr std::size_t bf16_row_bytes=4ULL*256*sizeof(std::uint16_t);
            constexpr std::size_t code_row_bytes=4ULL*64;
            constexpr std::size_t meta_row_bytes=4ULL*4*sizeof(float);
            std::size_t offset=0;
            const auto read_u32=[&](std::uint32_t& value) {
                require(offset+sizeof(value)<=live.size(),
                        "OSCAR eager live-state export truncated");
                std::memcpy(&value,live.data()+offset,sizeof(value));
                offset+=sizeof(value);
            };
            for (int bank=0;bank<16;++bank) {
                std::uint32_t model_layer=0,context=0,prefix=0,history=0,recent=0,head=0;
                read_u32(model_layer);read_u32(context);read_u32(prefix);
                read_u32(history);read_u32(recent);read_u32(head);
                require(model_layer==static_cast<std::uint32_t>(3+4*bank) &&
                            prefix<=64 && recent<=256 &&
                            head<256,
                        "OSCAR eager live-state scalar mismatch");
                require(context == static_cast<std::uint32_t>(
                            model_layer==3?expected_context:0),
                        "OSCAR eager host mirror advanced the wrong layer");
                const std::size_t payload=2*bf16_row_bytes*(prefix+recent)+
                    2*(code_row_bytes+meta_row_bytes)*history;
                require(offset+payload<=live.size(),
                        "OSCAR eager live-state payload truncated");
                offset+=payload;
            }
            require(offset==live.size(),"OSCAR eager live-state export has trailing bytes");
        };
        int eager_exact_cases=0;
        int eager_fallback_cases=0;
        for (const int rows : eager_rows) {
            for (const int base : eager_bases) {
                sequential->reset();
                cohort->reset();
                append_prefix(*sequential,base);
                append_prefix(*cohort,base);
                cuda_check(cudaMemcpyAsync(d_sequential_position,&base,sizeof(base),
                                            cudaMemcpyHostToDevice,stream),
                           "eager sequential position upload");
                cuda_check(cudaMemcpyAsync(d_cohort_position,&base,sizeof(base),
                                            cudaMemcpyHostToDevice,stream),
                           "eager cohort position upload");
                const int split=split_for_total(base+1);
                const bool same_class=split_for_total(base+1)==split_for_total(base+rows);
                const auto dispatch_before=cohort_telemetry.oscar_dispatches[3];
                const auto append_before=cohort_telemetry.append_calls;
                const auto aging_before=cohort_telemetry.aging_events;
                const auto boundary_before=
                    cohort_telemetry.continuation_cohort_eager_boundary_fallbacks;
                sequential->forward_continuation_device(
                    3,d_cohort_q,
                    d_cohort_k+static_cast<std::size_t>(base)*kKVRowValues,
                    d_cohort_v+static_cast<std::size_t>(base)*kKVRowValues,
                    d_sequential_out,rows,d_sequential_position,split,stream);
                const bool selected=cohort->try_forward_continuation_device_cohort_eager(
                    3,d_cohort_q,
                    d_cohort_k+static_cast<std::size_t>(base)*kKVRowValues,
                    d_cohort_v+static_cast<std::size_t>(base)*kKVRowValues,
                    d_cohort_out,rows,base,d_cohort_position,stream);
                cuda_check(cudaStreamSynchronize(stream),"eager cohort exact sync");
                require(selected==same_class,
                        "OSCAR eager cohort route mismatch at base " +
                            std::to_string(base) + " rows " + std::to_string(rows));
                if (same_class) {
                    ++eager_exact_cases;
                    const std::size_t output_bytes=static_cast<std::size_t>(rows)*
                        kQRowValues*sizeof(std::uint16_t);
                    require(download_bytes(d_sequential_out,output_bytes,
                                           "eager sequential output download") ==
                                download_bytes(d_cohort_out,output_bytes,
                                               "eager cohort output download"),
                            "OSCAR eager cohort output mismatch at base " +
                                std::to_string(base) + " rows " + std::to_string(rows));
                    sequential->set_graph_class(split);
                    cohort->set_graph_class(split);
                    require(sequential->graph_layer_state_host_for_test(3,stream) ==
                                cohort->graph_layer_state_host_for_test(3,stream),
                            "OSCAR eager cohort state mismatch at base " +
                                std::to_string(base) + " rows " + std::to_string(rows));
                    sequential->set_graph_class(0);
                    cohort->set_graph_class(0);
                    assert_eager_live_layer(base+rows);
                    const auto expected_aging=static_cast<std::uint64_t>(
                        eager_recent_begin(base+rows)-eager_recent_begin(base));
                    require(cohort_telemetry.oscar_dispatches[3] ==
                                dispatch_before+static_cast<std::uint64_t>(rows) &&
                            cohort_telemetry.append_calls ==
                                append_before+static_cast<std::uint64_t>(rows) &&
                            cohort_telemetry.aging_events == aging_before+expected_aging &&
                            cohort_telemetry.continuation_cohort_eager_dispatches ==
                                eager_dispatches_before+static_cast<std::uint64_t>(eager_exact_cases),
                            "OSCAR eager cohort telemetry publication mismatch");
                    require(cohort_telemetry.continuation_cohort_eager_boundary_fallbacks ==
                                boundary_before,
                            "OSCAR eager stable cohort emitted boundary fallback");
                } else {
                    ++eager_fallback_cases;
                    require(cohort_telemetry.oscar_dispatches[3] == dispatch_before &&
                            cohort_telemetry.append_calls == append_before &&
                            cohort_telemetry.aging_events == aging_before &&
                            cohort_telemetry.continuation_cohort_eager_dispatches ==
                                eager_dispatches_before+static_cast<std::uint64_t>(eager_exact_cases),
                            "OSCAR eager boundary fallback published cohort telemetry");
                    require(cohort_telemetry.continuation_cohort_eager_boundary_fallbacks ==
                                boundary_before+1,
                            "OSCAR eager boundary fallback was not observable");
                }
            }
        }
        require(cohort_telemetry.continuation_cohort_eager_attempts ==
                    eager_attempts_before+static_cast<std::uint64_t>(eager_rows.size()*eager_bases.size()) &&
                cohort_telemetry.continuation_cohort_eager_dispatches ==
                    eager_dispatches_before+static_cast<std::uint64_t>(eager_exact_cases) &&
                cohort_telemetry.continuation_cohort_eager_boundary_fallbacks ==
                    eager_boundary_before+static_cast<std::uint64_t>(eager_fallback_cases) &&
                cohort_telemetry.continuation_cohort_eager_latch_misses == eager_latch_before &&
                cohort_telemetry.continuation_cohort_eager_malformed == eager_malformed_before &&
                eager_exact_cases == 6 && eager_fallback_cases == 6,
                "OSCAR eager cohort route/counter boundary gate mismatch");
        std::cout << "OSCAR eager cohort exact gate: PASS rows=2/4/8 bases=511/512/8191/8192"
                  << " exact_cases=" << eager_exact_cases
                  << " boundary_fallbacks=" << eager_fallback_cases << "\n";

        // Matched same-stream whole-operator hook. Restore and position upload
        // are outside the event interval; both arms time only B8 OSCAR work.
        constexpr int kTimingReplays=100;
        constexpr int kTimingBase=4096;
        sequential->set_graph_class(0);
        cohort->set_graph_class(0);
        sequential->reset();
        cohort->reset();
        append_prefix(*sequential,kTimingBase);
        append_prefix(*cohort,kTimingBase);
        const std::size_t timing_checkpoint_bytes=sequential->checkpoint_device_bytes();
        void* d_timing_sequential=nullptr;
        void* d_timing_cohort=nullptr;
        cuda_check(cudaMalloc(&d_timing_sequential,timing_checkpoint_bytes),
                   "cohort timing sequential checkpoint alloc");
        cuda_check(cudaMalloc(&d_timing_cohort,timing_checkpoint_bytes),
                   "cohort timing checkpoint alloc");
        Exl3OscarCheckpoint timing_sequential{
            d_timing_sequential,timing_checkpoint_bytes};
        Exl3OscarCheckpoint timing_cohort{d_timing_cohort,timing_checkpoint_bytes};
        sequential->save_checkpoint(timing_sequential,stream);
        cohort->save_checkpoint(timing_cohort,stream);
        cuda_check(cudaStreamSynchronize(stream),"cohort timing checkpoint sync");
        cudaEvent_t timing_start=nullptr,timing_stop=nullptr;
        cuda_check(cudaEventCreate(&timing_start),"cohort timing start event");
        cuda_check(cudaEventCreate(&timing_stop),"cohort timing stop event");
        const auto time_arm=[&](Exl3OscarContext& context,
                                Exl3OscarCheckpoint& checkpoint,bool use_cohort) {
            float total_ms=0.0F;
            for (int replay=0;replay<kTimingReplays;++replay) {
                context.set_graph_class(0);
                context.restore_checkpoint(checkpoint,stream);
                cuda_check(cudaMemcpyAsync(use_cohort?d_cohort_position:d_sequential_position,
                                            &kTimingBase,sizeof(kTimingBase),
                                            cudaMemcpyHostToDevice,stream),
                           "cohort timing position upload");
                context.set_graph_class(32);
                cuda_check(cudaEventRecord(timing_start,stream),
                           "cohort timing start record");
                if (use_cohort) {
                    context.forward_continuation_device_cohort_b8(
                        3,d_cohort_q,
                        d_cohort_k+static_cast<std::size_t>(kTimingBase)*kKVRowValues,
                        d_cohort_v+static_cast<std::size_t>(kTimingBase)*kKVRowValues,
                        d_cohort_out,d_cohort_position,32,stream);
                } else {
                    context.forward_continuation_device(
                        3,d_cohort_q,
                        d_cohort_k+static_cast<std::size_t>(kTimingBase)*kKVRowValues,
                        d_cohort_v+static_cast<std::size_t>(kTimingBase)*kKVRowValues,
                        d_sequential_out,kCohortRows,d_sequential_position,32,stream);
                }
                cuda_check(cudaEventRecord(timing_stop,stream),
                           "cohort timing stop record");
                cuda_check(cudaEventSynchronize(timing_stop),
                           "cohort timing stop sync");
                float elapsed=0.0F;
                cuda_check(cudaEventElapsedTime(&elapsed,timing_start,timing_stop),
                           "cohort timing elapsed");
                total_ms+=elapsed;
            }
            context.set_graph_class(0);
            return total_ms/static_cast<float>(kTimingReplays);
        };
        const float sequential_ms=time_arm(*sequential,timing_sequential,false);
        const float cohort_ms=time_arm(*cohort,timing_cohort,true);
        const float whole_gain=100.0F*(sequential_ms-cohort_ms)/sequential_ms;
        std::cout << "OSCAR cohort B8 operator: PASS cases=" << bases.size()
                  << " timing_replays=" << kTimingReplays
                  << " sequential_ms=" << sequential_ms
                  << " cohort_ms=" << cohort_ms
                  << " whole_gain_pct=" << whole_gain << "\n";
        cudaEventDestroy(timing_start);
        cudaEventDestroy(timing_stop);
        cudaFree(d_timing_sequential);
        cudaFree(d_timing_cohort);
        cudaFree(d_cohort_q);
        cudaFree(d_cohort_k);
        cudaFree(d_cohort_v);
        cudaFree(d_sequential_out);
        cudaFree(d_cohort_out);
        cudaFree(d_sequential_position);
        cudaFree(d_cohort_position);
    }

    cudaFree(d_rotated_k);
    cudaFree(d_rotated_v);
    cudaFree(d_checkpoint_a);
    cudaFree(d_checkpoint_b);
    cudaFree(d_reference_out);
    cudaFree(d_q);
    cudaFree(d_out);
    cudaFree(d_qtaps);
    cudaFree(d_ktaps);
    cudaFree(d_vtaps);
    cuda_check(cudaStreamDestroy(stream), "adapter test stream destroy");
    std::cout << "E4C1 EXL3 OSCAR adapter test: PASS\n";
    return 0;
}

}  // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& error) {
        std::cerr << "E4C1 EXL3 OSCAR adapter test: FAIL: " << error.what() << "\n";
        return 1;
    }
}
