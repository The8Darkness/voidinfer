// Diagnostic probe: native MTP next-next-token accuracy along a given exact
// target sequence (prompt + verified greedy continuation). For every position
// p it teacher-forces hidden[p] + token[p+1] through the native MTP module and
// records the rank of the true token[p+2] among the MTP logits (top-16
// retained) for continuation targets. Target authority is untouched; the CSV
// is joined offline with DFlash2 round logs.
#include "exl3/text_model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

std::vector<std::int64_t> load_ids(const char* path) {
    std::ifstream input(path ? path : "");
    require(input.good(), std::string("MTP probe id file unavailable: ") + (path ? path : ""));
    std::vector<std::int64_t> result;
    std::int64_t token = 0;
    while (input >> token) result.push_back(token);
    require(!result.empty(), "MTP probe id file is empty");
    return result;
}

std::vector<int> top_indices(const std::vector<std::uint16_t>& logits, int count) {
    std::vector<int> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    auto value = [&](int i) { return __half2float(__ushort_as_half(logits[i])); };
    std::partial_sort(order.begin(), order.begin() + count, order.end(), [&](int a, int b) {
        const float va = value(a), vb = value(b);
        return va > vb || (va == vb && a < b);
    });
    order.resize(count);
    return order;
}

} // namespace

int main() {
    try {
        const char* target = std::getenv("NINFER_EXL3_TARGET_PATH");
        const char* prompt = std::getenv("NINFER_MTP_PROBE_PROMPT");
        const char* continuation = std::getenv("NINFER_MTP_PROBE_CONTINUATION");
        const char* out_path = std::getenv("NINFER_MTP_PROBE_OUT");
        if (!target || !*target || !prompt || !*prompt || !continuation || !*continuation ||
            !out_path || !*out_path) {
            std::cerr << "MTP accuracy probe skipped: target/prompt/continuation/output missing\n";
            return 77;
        }
        const auto prompt_ids = load_ids(prompt);
        const auto tail = load_ids(continuation);
        std::vector<std::int64_t> sequence(prompt_ids);
        sequence.insert(sequence.end(), tail.begin(), tail.end());
        const int prompt_rows = static_cast<int>(prompt_ids.size());
        const int total = static_cast<int>(sequence.size());

        auto model = ninfer::exl3::Exl3TextModel::load(target, total + 16);
        auto context = model->create_context(false);
        context->prepare_native_mtp_hidden_capture(total + 8);
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "probe stream");

        std::shared_ptr<ninfer::exl3::Exl3MtpPrefixState> mtp;
        std::ofstream out(out_path);
        out << "position,true_token,mtp_top1,true_rank,top16\n";
        std::vector<std::uint16_t> logits_host(ninfer::exl3::kExl3MtpVocabSize);
        int mtp_rows = 0, evaluated = 0, top1 = 0, top4 = 0;
        // Row p consumes hidden[p] and token[p+1]; its logits predict token[p+2].
        const auto advance_mtp = [&](int hidden_available) {
            cuda_check(cudaDeviceSynchronize(), "probe target drain");
            if (!mtp) {
                mtp = context->create_native_mtp_prefix_state(static_cast<std::uint32_t>(total + 8));
                require(mtp && mtp->valid(), "native MTP state unavailable");
            }
            for (; mtp_rows < hidden_available && mtp_rows + 1 < total; ++mtp_rows) {
                const int p = mtp_rows;
                const auto rows = context->target_hidden_handoffs(p, 1);
                require(rows.size() == 1 && rows.front().position == p, "MTP probe hidden row");
                const std::int64_t tokens[2] = {sequence[p], sequence[p + 1]};
                const auto receipt = mtp->prefill_target_prefix(
                    rows, std::span<const std::int64_t>(tokens, 2), 1, 1,
                    reinterpret_cast<std::uintptr_t>(stream));
                require(receipt.valid(), "MTP probe step failed at position " + std::to_string(p));
                if (p + 2 < prompt_rows || p + 2 >= total) continue;
                cuda_check(cudaStreamSynchronize(stream), "MTP probe step drain");
                const auto view = mtp->last_logits();
                cuda_check(cudaMemcpy(logits_host.data(), reinterpret_cast<const void*>(view.data),
                                      logits_host.size() * sizeof(std::uint16_t),
                                      cudaMemcpyDeviceToHost),
                           "MTP probe logits copy");
                const auto top = top_indices(logits_host, 16);
                const auto truth = sequence[p + 2];
                int rank = -1;
                for (int i = 0; i < 16; ++i)
                    if (top[i] == truth) { rank = i; break; }
                ++evaluated;
                top1 += rank == 0;
                top4 += rank >= 0 && rank < 4;
                out << p << ',' << truth << ',' << top[0] << ',' << rank << ',';
                for (int i = 0; i < 16; ++i) out << top[i] << (i + 1 < 16 ? ' ' : '\n');
            }
        };

        const int first = std::min(16, total);
        context->prefill(std::span<const std::int64_t>(sequence.data(), first));
        advance_mtp(first);
        std::cout << "MTP_PROBE first rows ok" << std::endl;
        bool wide = true;
        for (int offset = first; offset < total;) {
            int count = std::min(wide ? 16 : 8, total - offset);
            try {
                if (wide) context->append_prefill_wide(
                    std::span<const std::int64_t>(sequence.data() + offset, count));
                else context->append_prefill(
                    std::span<const std::int64_t>(sequence.data() + offset, count));
            } catch (const std::exception& error) {
                require(wide && offset == first,
                        std::string("MTP probe append failed: ") + error.what());
                std::cout << "MTP_PROBE wide append unavailable (" << error.what()
                          << "); using 8-row chunks" << std::endl;
                wide = false;
                continue;
            }
            offset += count;
            advance_mtp(offset);
        }
        std::cout << "MTP_PROBE PASS prompt=" << prompt_rows << " continuation=" << tail.size()
                  << " evaluated=" << evaluated << " top1=" << top1 << " top4=" << top4 << '\n';
        std::ofstream seq(std::string(out_path) + ".sequence");
        for (auto token : sequence) seq << token << '\n';
        cudaStreamDestroy(stream);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
