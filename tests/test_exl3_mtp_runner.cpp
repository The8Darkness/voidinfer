#include "exl3/mtp_runner.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

int check(bool value, const char* message) {
    if (value) return 0;
    std::cerr << message << '\n';
    return 1;
}

struct CallbackObservation {
    int calls = 0;
    bool input_aligned = false;
    bool binding_aligned = false;
    std::uintptr_t stream = 0;
};

bool observe_enqueue(const ninfer::exl3::Exl3MtpExecutionBinding& binding,
                     const ninfer::exl3::Exl3MtpStepInput& input,
                     const ninfer::exl3::Exl3MtpStepOutput& output,
                     std::uintptr_t stream, void* opaque) noexcept {
    auto& observation = *static_cast<CallbackObservation*>(opaque);
    ++observation.calls;
    observation.input_aligned =
        input.seed_kind == ninfer::exl3::Exl3MtpSeedKind::TargetHidden && input.position == 17 &&
        input.token_position == 18 && input.token == 1234 &&
        input.hidden.rows == ninfer::exl3::kExl3MtpHiddenWidth &&
        input.hidden.cols == 1 && output.hidden.rows == ninfer::exl3::kExl3MtpHiddenWidth &&
        output.logits.rows == ninfer::exl3::kExl3MtpVocabSize &&
        output.predicted_token.rows == 1;
    observation.binding_aligned = binding.valid();
    observation.stream = stream;
    return true;
}

bool reject_enqueue(const ninfer::exl3::Exl3MtpExecutionBinding&,
                    const ninfer::exl3::Exl3MtpStepInput&,
                    const ninfer::exl3::Exl3MtpStepOutput&, std::uintptr_t, void*) noexcept {
    return false;
}

struct BindingFixture {
    ninfer::exl3::Exl3MtpBinding manifest;
    std::vector<ninfer::exl3::Exl3MtpDeviceTensor> tensors;
    std::shared_ptr<int> manifest_owner = std::make_shared<int>(1);
    std::shared_ptr<int> target_owner   = std::make_shared<int>(2);
    std::shared_ptr<int> kv_owner      = std::make_shared<int>(3);
    std::shared_ptr<int> stream_owner  = std::make_shared<int>(4);

    ninfer::exl3::Exl3MtpExecutionBinding binding() {
        const auto specs = ninfer::exl3::qwen3_8_27b_native_mtp_manifest();
        manifest.total_bytes = ninfer::exl3::kQwen38NativeMtpTotalBytes;
        manifest.tensors.reserve(specs.size());
        tensors.reserve(specs.size());
        for (const auto& spec : specs) {
            ninfer::exl3::TensorInfo info;
            info.name       = std::string(spec.name);
            info.dtype      = std::string(spec.dtype);
            info.shape.assign(spec.shape.begin(), spec.shape.begin() + spec.rank);
            info.data_end   = spec.bytes;
            manifest.tensors.push_back({spec.name, {}, std::move(info)});
            tensors.push_back({spec.name, static_cast<std::uintptr_t>(0x1000 + tensors.size()),
                               static_cast<std::size_t>(spec.bytes)});
        }
        ninfer::exl3::Exl3MtpExecutionBinding result{
            .manifest       = &manifest,
            .tensors        = tensors,
            .shared_target  = {{0x2000, 1, 1, 1}, {0x3000, 1, 1, 1}, target_owner},
            .weights_owner = manifest_owner,
            .model_identity = 42};
        return result;
    }
};

ninfer::exl3::Exl3MtpStepInput input(ninfer::exl3::Exl3MtpSeedKind seed_kind =
                                         ninfer::exl3::Exl3MtpSeedKind::TargetHidden) {
    return {.request_identity = 11,
            .target_identity  = 22,
            .model_identity   = 42,
            .hidden_generation = 9,
            .position          = 17,
            .token_position    = 18,
            .token             = 1234,
            .seed_kind         = seed_kind,
            .hidden_owner      = std::make_shared<int>(17),
            .hidden            = {0x4000, 10'240, ninfer::exl3::kExl3MtpHiddenWidth, 1}};
}

ninfer::exl3::Exl3MtpStepOutput output() {
    return {.hidden = {0x5000, 10'240, ninfer::exl3::kExl3MtpHiddenWidth, 1},
            .logits = {0x6000, 496'640, ninfer::exl3::kExl3MtpVocabSize, 1},
            .predicted_token = {0x7000, sizeof(std::int32_t), 1, 1}};
}

} // namespace

int main() {
    using namespace ninfer::exl3;
    int failures = 0;

    BindingFixture fixture;
    auto binding = fixture.binding();
    failures += check(binding.valid(), "strict 39-tensor MTP execution binding was rejected");

    auto no_executor = std::make_shared<int>(5);
    Exl3MtpOneStepRunner missing_executor(binding, fixture.kv_owner, fixture.stream_owner);
    failures += check(!missing_executor.submit(input(), output(), no_executor, 0x81, 0x91),
                      "missing MTP executor did not fail closed");
    failures += check(missing_executor.phase() == Exl3MtpRunnerPhase::Failed &&
                          missing_executor.fallback_reason() == Exl3MtpFallbackReason::MissingExecutor,
                      "missing executor fallback state was not retained");
    failures += check(missing_executor.reset() && missing_executor.phase() == Exl3MtpRunnerPhase::Idle,
                      "failed MTP runner did not reset cleanly");

    CallbackObservation observed;
    Exl3MtpOneStepRunner runner(binding, fixture.kv_owner, fixture.stream_owner,
                                &observe_enqueue, &observed);
    auto output_owner = std::make_shared<int>(6);
    std::weak_ptr<const void> output_weak = output_owner;
    const auto ticket = runner.submit(input(), output(), output_owner, 0x81, 0x91);
    output_owner.reset();
    failures += check(ticket && runner.phase() == Exl3MtpRunnerPhase::Submitted && observed.calls == 1,
                      "one-step MTP submission did not retain its owned ticket");
    failures += check(observed.input_aligned && observed.binding_aligned && observed.stream == 0x81,
                      "one-step callback did not receive target-hidden/token/position contract");
    failures += check(!output_weak.expired(), "MTP output owner was released before final use");
    failures += check(runner.output_for_test() == nullptr,
                      "MTP output became visible before producer event");
    failures += check(!runner.producer_finished(ticket, 0x92),
                      "foreign producer event was accepted");
    failures += check(runner.phase() == Exl3MtpRunnerPhase::Submitted &&
                          runner.fallback_reason() == Exl3MtpFallbackReason::EventMismatch,
                      "foreign producer event abandoned the live submission");
    failures += check(!runner.reset() && !output_weak.expired(),
                      "live MTP output owner retired without producer/final-use witnesses");
    failures += check(runner.producer_finished(ticket, 0x91) &&
                          runner.final_use(ticket, 0xA1) && runner.reset() &&
                          output_weak.expired(),
                      "live MTP output did not retire after exact lifetime witnesses");

    observed = {};
    const auto second = runner.submit(input(), output(), std::make_shared<int>(7), 0x81, 0x91);
    failures += check(runner.producer_finished(second, 0x91) &&
                          runner.phase() == Exl3MtpRunnerPhase::Ready &&
                          runner.output_for_test() != nullptr,
                      "matching producer event did not publish one-step output");
    failures += check(!runner.reset() &&
                          runner.phase() == Exl3MtpRunnerPhase::Ready &&
                          runner.fallback_reason() == Exl3MtpFallbackReason::MissingFinalUse,
                      "live MTP output reset without a final-use witness");
    failures += check(!runner.final_use(second, 0x91),
                      "producer event was reused as final-use witness");
    failures += check(runner.final_use(second, 0xA1) && runner.phase() == Exl3MtpRunnerPhase::Retired,
                      "distinct final-use event did not retire MTP output");
    failures += check(runner.reset() && runner.phase() == Exl3MtpRunnerPhase::Idle,
                      "retired MTP step did not reset");

    // A chained step is accepted only with an explicit prior-MTP seed.  This
    // is the future DFlash2->MTP suffix path; it never fabricates target_hidden.
    const auto chained = runner.submit(input(Exl3MtpSeedKind::PreviousMtpHidden), output(),
                                       std::make_shared<int>(8), 0x82, 0xA2);
    failures += check(chained && runner.input_for_test()->seed_kind == Exl3MtpSeedKind::PreviousMtpHidden,
                      "teacher-forced MTP chain input was not represented explicitly");
    failures += check(runner.producer_finished(chained, 0xA2) && runner.final_use(chained, 0xB2) &&
                          runner.reset(),
                      "teacher-forced MTP chain could not complete ownership lifecycle");

    Exl3MtpOneStepRunner rejected(binding, fixture.kv_owner, fixture.stream_owner,
                                  &reject_enqueue, nullptr);
    const auto rejected_ticket = rejected.submit(input(), output(), std::make_shared<int>(9), 0x83,
                                                 0x93);
    failures += check(!rejected_ticket && rejected.phase() == Exl3MtpRunnerPhase::Failed &&
                          rejected.fallback_reason() == Exl3MtpFallbackReason::ExecutionFailed,
                      "executor rejection did not force ordinary-target fallback");

    auto wrong_model = input();
    wrong_model.model_identity = 99;
    failures += check(!runner.submit(wrong_model, output(), std::make_shared<int>(10), 0x84, 0x94) &&
                          runner.fallback_reason() == Exl3MtpFallbackReason::IdentityMismatch,
                      "model identity mismatch was not rejected");

    if (failures != 0) return 1;
    std::cout << "exl3_mtp_runner_contract: PASS\n";
    return 0;
}
