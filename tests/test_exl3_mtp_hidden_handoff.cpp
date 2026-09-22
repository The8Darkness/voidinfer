#include "exl3/text_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
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
    require(input.good(), "native MTP handoff prompt is unavailable");
    std::vector<std::int64_t> result;
    std::int64_t token = 0;
    while (input >> token) result.push_back(token);
    require(result.size() >= 14, "native MTP handoff prompt needs 14 tokens");
    result.resize(14);
    return result;
}

} // namespace

int main() {
    try {
        const char* target = std::getenv("NINFER_EXL3_TARGET_PATH");
        const char* prompt = std::getenv("NINFER_MTP_HANDOFF_PROMPT_FILE");
        if (!target || !*target || !prompt || !*prompt) {
            std::cerr << "native MTP hidden handoff skipped: target/prompt missing\n";
            return 77;
        }
        auto model = ninfer::exl3::Exl3TextModel::load(target, 64);
        auto context = model->create_context(false);
        context->prepare_native_mtp_hidden_capture(64);
        const auto ids = load_ids(prompt);
        context->prefill(ids);
        require(context->native_mtp_hidden_capture_prepared() &&
                    context->native_mtp_hidden_capture_rows() == 14 &&
                    context->native_mtp_hidden_capture_first_position() == 0,
                "native MTP hidden capture did not retain the complete prompt interval");
        const auto prompt_rows = context->target_hidden_handoffs(0, 14);
        require(prompt_rows.size() == 14 && prompt_rows.front().position == 0 &&
                    prompt_rows.back().position == 13,
                "native MTP hidden range did not preserve chronological positions");
        auto first = context->target_hidden_handoff(13);
        require(first.valid() && first.position == 13,
                "native MTP handoff did not publish the authoritative prompt-tail position");
        std::vector<std::uint16_t> before(ninfer::exl3::kExl3MtpHiddenWidth);
        cuda_check(cudaMemcpy(before.data(), reinterpret_cast<const void*>(first.hidden.data),
                              first.hidden.bytes, cudaMemcpyDeviceToHost),
                   "copy native MTP hidden before target reuse");

        const auto logits = context->logits_host();
        const auto next = static_cast<std::int64_t>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        context->decode(next);

        std::vector<std::uint16_t> after(before.size());
        cuda_check(cudaMemcpy(after.data(), reinterpret_cast<const void*>(first.hidden.data),
                              first.hidden.bytes, cudaMemcpyDeviceToHost),
                   "copy native MTP hidden after target reuse");
        require(before == after,
                "owned native MTP hidden changed after target ping-pong buffer reuse");
        const auto second = context->target_hidden_handoff(14);
        require(second.valid() && second.position == 14 &&
                    second.hidden_generation != first.hidden_generation &&
                    second.model_identity == first.model_identity,
                "native MTP hidden generation/position/model identity did not advance exactly");
        std::cout << "exl3_mtp_hidden_handoff: PASS prompt_rows=14 positions=0..14"
                     " full_capture=1 retained_across_decode=1\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
