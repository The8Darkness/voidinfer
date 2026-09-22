#pragma once

#include "exl3/linear_cuda.h"
#include "exl3/mtp_manifest.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <span>
#include <string_view>
#include <utility>

namespace ninfer::exl3 {

// The native Qwen3.8-27B MTP dimensions are part of the closed manifest.  Keep
// these values in the EXL3 bridge so a caller cannot silently pass a DFlash
// feature row, a target logit row, or a hidden row from another model family.
inline constexpr std::uint32_t kExl3MtpHiddenWidth = 5'120;
// The MTP checkpoint has no private embedding/head.  This bridge therefore
// uses the pinned target's full 248,320-row H6 vocabulary; the optional native
// 131,072-row shortlist is a separate proposal-head optimization and cannot
// silently replace target verification semantics.
inline constexpr std::uint32_t kExl3MtpVocabSize   = 248'320;
// The physical LM-head row count includes unused padding rows.  Proposals
// must remain in the target tokenizer domain, as in the native target path.
inline constexpr std::uint32_t kExl3MtpTokenDomain = 248'077;
inline constexpr std::uint32_t kExl3MtpOneLayer    = 1;

// An opaque device allocation/view.  The runner deliberately does not own or
// reinterpret the allocation: the execution callback owns the EXL3/K4 ops and
// receives these explicit views.  Rows/columns describe a column-major Tensor
// view, matching the native MTP schedule's [hidden,T] convention.
struct Exl3MtpBuffer {
    std::uintptr_t data = 0;
    std::size_t bytes    = 0;
    std::uint32_t rows   = 0;
    std::uint32_t cols   = 0;

    [[nodiscard]] bool valid() const noexcept {
        return data != 0 && bytes != 0 && rows != 0 && cols != 0;
    }

    [[nodiscard]] bool exact(std::uint32_t expected_rows,
                             std::uint32_t expected_cols,
                             std::size_t minimum_bytes) const noexcept {
        return valid() && rows == expected_rows && cols == expected_cols &&
               bytes >= minimum_bytes;
    }
};

// Owned copy of the target's final pre-LM-head hidden row.  The native MTP
// bridge consumes this exact target hidden (not a tap or a recomputed proxy),
// so Exl3TextContext copies it into a separate device allocation before this
// handoff is published.  The owner keeps the copy live across asynchronous
// MTP work even if the target context reuses its ping-pong hidden buffers.
struct Exl3MtpTargetHiddenHandoff {
    std::shared_ptr<const void> owner;
    std::shared_ptr<const void> model_identity;
    Exl3MtpBuffer hidden;
    std::uint64_t hidden_generation = 0;
    std::int64_t position = -1;

    [[nodiscard]] bool valid() const noexcept {
        return owner && model_identity && hidden_generation != 0 && position >= 0 &&
            hidden.exact(kExl3MtpHiddenWidth, 1,
                         static_cast<std::size_t>(kExl3MtpHiddenWidth) * 2U);
    }
};

// The manifest is metadata-only.  Materialized native EXL3 tensors are passed
// separately and must retain a descriptor name/byte extent for every one of the
// 39 manifest entries.  This keeps materialization and the K4 operator binding
// explicit; no TextContext or hidden model view is borrowed by this runner.
struct Exl3MtpDeviceTensor {
    std::string_view name;
    std::uintptr_t data = 0;
    std::size_t bytes    = 0;
};

struct Exl3MtpSharedTargetBinding {
    // The native MTP stem reuses target embedding; proposal selection reuses
    // target lm_head.  Both are explicit bindings, not implicit model globals.
    Exl3MtpBuffer token_embedding;
    Exl3MtpBuffer lm_head;
    // The raw buffer identity above keeps lifetime explicit. These immutable
    // descriptors bind the target's actual H6 EXL3 head; an MTP checkpoint
    // does not contain a private proposal head.
    std::shared_ptr<const void> owner;
    // Stable identity token used to match target-hidden handoffs. `owner`
    // may be a stronger model-backing owner than this token and is retained
    // for the complete asynchronous MTP lifetime.
    std::shared_ptr<const void> model_identity;
    Exl3CudaLinearWeights lm_head_weights{};
    Exl3CudaLinearMetadata lm_head_metadata{};

    [[nodiscard]] bool valid() const noexcept {
        return token_embedding.valid() && lm_head.valid() && owner != nullptr &&
            model_identity != nullptr;
    }
};

// Optional pre-populated one-layer MTP KV. Rows are chronological and each
// row stores [4 KV heads, 256 dimensions] in represented FP16. The bridge
// writes row `input.position`; valid_rows must equal that position. A real
// bridge at a nonzero position is invalid without this prefix-populated cache;
// self-row attention is not a substitute for the missing causal prefix.
struct Exl3MtpKvBinding {
    Exl3MtpBuffer key;
    Exl3MtpBuffer value;
    std::uint32_t capacity = 0;
    std::uint32_t valid_rows = 0;
    std::shared_ptr<const void> owner;

    [[nodiscard]] bool valid() const noexcept {
        const auto row_bytes = static_cast<std::size_t>(1'024U) * 2U;
        return owner && capacity != 0 && valid_rows <= capacity && key.data != 0 &&
            value.data != 0 && key.rows == capacity && value.rows == capacity &&
            key.cols == 1'024U && value.cols == 1'024U &&
            key.bytes >= static_cast<std::size_t>(capacity) * row_bytes &&
            value.bytes >= static_cast<std::size_t>(capacity) * row_bytes;
    }
};

struct Exl3MtpExecutionBinding {
    const Exl3MtpBinding* manifest = nullptr;
    std::span<const Exl3MtpDeviceTensor> tensors;
    Exl3MtpSharedTargetBinding shared_target;
    Exl3MtpKvBinding mtp_kv;
    // This owner retains the materialized 39-tensor view while an asynchronous
    // callback is using it.  A raw manifest pointer alone is not sufficient.
    std::shared_ptr<const void> weights_owner;
    std::uint64_t model_identity = 0;

    [[nodiscard]] bool valid() const noexcept {
        if (manifest == nullptr || manifest->tensors.size() != kQwen38NativeMtpTensorCount ||
            manifest->total_bytes != kQwen38NativeMtpTotalBytes || tensors.size() !=
                kQwen38NativeMtpTensorCount || model_identity == 0 || !weights_owner ||
            !shared_target.valid()) {
            return false;
        }

        const auto specs = qwen3_8_27b_native_mtp_manifest();
        if (specs.size() != kQwen38NativeMtpTensorCount) return false;
        for (const auto& spec : specs) {
            const auto* descriptor = manifest->find(spec.name);
            if (descriptor == nullptr || descriptor->info.bytes() != spec.bytes ||
                descriptor->info.dtype != spec.dtype || descriptor->info.shape.size() != spec.rank ||
                !std::equal(descriptor->info.shape.begin(), descriptor->info.shape.end(),
                            spec.shape.begin())) {
                return false;
            }
            bool found = false;
            for (const auto& tensor : tensors) {
                if (tensor.name != spec.name) continue;
                if (found || tensor.data == 0 || tensor.bytes != spec.bytes) return false;
                found = true;
            }
            if (!found) return false;
        }
        return true;
    }
};

// Completion witness for a bounded target-prefix MTP population.  The
// producer event is recorded after every requested hidden[p]+token[p+1] row
// has been enqueued.  The owner is retained by the caller until the event has
// been consumed; a raw pointer or a stream identity alone is not sufficient
// to keep the MTP KV and intermediate hidden row live.
struct Exl3MtpPrefixReceipt {
    std::shared_ptr<const void> owner;
    std::uint64_t request_identity = 0;
    std::uint64_t target_identity = 0;
    std::uint64_t model_identity = 0;
    std::uint64_t hidden_generation = 0;
    std::uint32_t valid_rows = 0;
    std::int64_t first_position = -1;
    std::int64_t last_position = -1;
    std::uintptr_t producer_event = 0;

    [[nodiscard]] bool valid() const noexcept {
        return owner && request_identity != 0 && target_identity != 0 &&
               model_identity != 0 && hidden_generation != 0 && valid_rows != 0 &&
               first_position >= 0 && last_position >= first_position &&
               producer_event != 0;
    }
};

enum class Exl3MtpSeedKind : std::uint8_t {
    TargetHidden,
    PreviousMtpHidden,
};

// One row in the chronological MTP chain.  For the first bridge, hidden is
// authoritative target_hidden[p] and token is token[p+1].  For later teacher-
// forced suffix steps, hidden is the prior MTP hidden and token is the supplied
// next token.  No API accepts a token without the corresponding hidden owner.
struct Exl3MtpStepInput {
    std::uint64_t request_identity = 0;
    std::uint64_t target_identity  = 0;
    std::uint64_t model_identity   = 0;
    std::uint64_t hidden_generation = 0;
    std::int64_t position           = -1;
    // The token consumed by the stem is the following sequence token.  Keep
    // this explicit so a caller cannot accidentally pass token[p] at position
    // p and still appear to satisfy the one-step bridge contract.
    std::int64_t token_position     = -1;
    std::int32_t token              = -1;
    Exl3MtpSeedKind seed_kind       = Exl3MtpSeedKind::TargetHidden;
    // Keeps target_hidden[p] or the prior MTP hidden alive until the producer
    // callback has consumed it.  A raw device pointer is never sufficient.
    std::shared_ptr<const void> hidden_owner;
    Exl3MtpBuffer hidden;

    [[nodiscard]] bool valid() const noexcept {
        if (request_identity == 0 || target_identity == 0 || model_identity == 0 ||
            hidden_generation == 0 || position < 0 ||
            position == std::numeric_limits<std::int64_t>::max() ||
            token_position != position + 1 ||
            token < 0 || !hidden_owner ||
            static_cast<std::uint32_t>(token) >= kExl3MtpTokenDomain) {
            return false;
        }
        return hidden.exact(kExl3MtpHiddenWidth, 1,
                            static_cast<std::size_t>(kExl3MtpHiddenWidth) * 2U);
    }
};

// The executor is the only place allowed to enqueue native EXL3 arithmetic.
// It must enqueue one MTP step on retained_stream and write output buffers. A
// false return means that the operation was not submitted and the runner must
// fall back to ordinary target verification. It must not partially publish an
// output on false.
struct Exl3MtpStepOutput {
    Exl3MtpBuffer hidden;
    Exl3MtpBuffer logits;
    Exl3MtpBuffer predicted_token;
};

using Exl3MtpEnqueueOneStep = bool (*)(const Exl3MtpExecutionBinding& binding,
                                       const Exl3MtpStepInput& input,
                                       const Exl3MtpStepOutput& output,
                                       std::uintptr_t retained_stream,
                                       void* user) noexcept;

enum class Exl3MtpRunnerPhase : std::uint8_t {
    Idle,
    Submitted,
    Ready,
    Failed,
    Retired,
};

enum class Exl3MtpFallbackReason : std::uint8_t {
    None,
    InvalidBinding,
    MissingExecutor,
    InvalidInput,
    InvalidOutput,
    Busy,
    IdentityMismatch,
    EventMismatch,
    ExecutionFailed,
    MissingFinalUse,
};

// A deliberately small, single-in-flight owner for one native MTP step.  The
// class is usable by a DFlash2-extension caller, but it is not a second
// speculative scheduler: target verification remains authoritative and this
// owner only exposes a prediction after its producer event has been witnessed.
class Exl3MtpOneStepRunner {
public:
    struct Ticket {
        std::uint64_t generation = 0;
        std::uint64_t request_identity = 0;

        [[nodiscard]] explicit operator bool() const noexcept {
            return generation != 0 && request_identity != 0;
        }
    };

    Exl3MtpOneStepRunner(Exl3MtpExecutionBinding binding,
                         std::shared_ptr<const void> mtp_kv_owner,
                         std::shared_ptr<const void> stream_owner,
                         Exl3MtpEnqueueOneStep enqueue = nullptr,
                         void* user = nullptr) noexcept
        : binding_(std::move(binding)),
          mtp_kv_owner_(std::move(mtp_kv_owner)),
          stream_owner_(std::move(stream_owner)),
          enqueue_(enqueue),
          user_(user) {}

    Exl3MtpOneStepRunner(const Exl3MtpOneStepRunner&)            = delete;
    Exl3MtpOneStepRunner& operator=(const Exl3MtpOneStepRunner&) = delete;

    [[nodiscard]] Exl3MtpRunnerPhase phase() const noexcept { return phase_; }
    [[nodiscard]] Exl3MtpFallbackReason fallback_reason() const noexcept { return reason_; }
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
    [[nodiscard]] const Exl3MtpStepInput* input_for_test() const noexcept {
        return phase_ == Exl3MtpRunnerPhase::Idle ? nullptr : &input_;
    }
    [[nodiscard]] const Exl3MtpStepOutput* output_for_test() const noexcept {
        return phase_ == Exl3MtpRunnerPhase::Ready ? &output_ : nullptr;
    }

    // Submit one step.  The caller supplies the stream/event identities and
    // output owner because only the CUDA owner knows their real lifetime.  A
    // false result is a clean fail-closed fallback; no asynchronous operation
    // is considered owned unless the callback returned true.
    [[nodiscard]] Ticket submit(const Exl3MtpStepInput& input,
                                Exl3MtpStepOutput output,
                                std::shared_ptr<const void> output_owner,
                                std::uintptr_t retained_stream,
                                std::uintptr_t producer_event) noexcept {
        if (phase_ != Exl3MtpRunnerPhase::Idle && phase_ != Exl3MtpRunnerPhase::Retired) {
            fail(Exl3MtpFallbackReason::Busy);
            return {};
        }
        clear_for_new_step();
        if (!binding_.valid()) {
            return fail_ticket(Exl3MtpFallbackReason::InvalidBinding);
        }
        if (enqueue_ == nullptr) {
            return fail_ticket(Exl3MtpFallbackReason::MissingExecutor);
        }
        if (!input.valid()) {
            return fail_ticket(Exl3MtpFallbackReason::InvalidInput);
        }
        if (input.model_identity != binding_.model_identity || !mtp_kv_owner_ ||
            !stream_owner_ || !output_owner || retained_stream == 0 || producer_event == 0) {
            return fail_ticket(input.model_identity != binding_.model_identity
                                   ? Exl3MtpFallbackReason::IdentityMismatch
                                   : Exl3MtpFallbackReason::InvalidOutput);
        }
        if (!output.hidden.exact(kExl3MtpHiddenWidth, 1,
                                 static_cast<std::size_t>(kExl3MtpHiddenWidth) * 2U) ||
            !output.logits.exact(kExl3MtpVocabSize, 1,
                                 static_cast<std::size_t>(kExl3MtpVocabSize) * 2U) ||
            !output.predicted_token.exact(1, 1, sizeof(std::int32_t))) {
            return fail_ticket(Exl3MtpFallbackReason::InvalidOutput);
        }

        input_ = input;
        output_ = output;
        output_owner_ = std::move(output_owner);
        retained_stream_ = retained_stream;
        producer_event_ = producer_event;
        final_use_event_ = 0;
        ++generation_;
        if (generation_ == 0) { // wraparound is never a valid alias
            return fail_ticket(Exl3MtpFallbackReason::ExecutionFailed);
        }
        if (!enqueue_(binding_, input_, output_, retained_stream_, user_)) {
            return fail_ticket(Exl3MtpFallbackReason::ExecutionFailed);
        }
        phase_ = Exl3MtpRunnerPhase::Submitted;
        reason_ = Exl3MtpFallbackReason::None;
        return {generation_, input_.request_identity};
    }

    // Witness the exact producer event.  A recycled or foreign event cannot
    // make output visible.  Errors are sticky and force ordinary target
    // fallback; output_owner_ remains retained until reset/retire.
    [[nodiscard]] bool producer_finished(Ticket ticket, std::uintptr_t event,
                                         int error = 0) noexcept {
        if (!matches(ticket) || phase_ != Exl3MtpRunnerPhase::Submitted ||
            event == 0 || event != producer_event_) {
            // Do not abandon a genuinely submitted asynchronous operation on
            // a bad witness.  Its owners must remain live until the exact
            // producer event is observed and the consumer records final use.
            reason_ = Exl3MtpFallbackReason::EventMismatch;
            return false;
        }
        if (error != 0) {
            fail(Exl3MtpFallbackReason::ExecutionFailed);
            return false;
        }
        phase_ = Exl3MtpRunnerPhase::Ready;
        reason_ = Exl3MtpFallbackReason::None;
        return true;
    }

    // Output is consumable only after producer_finished.  The caller must
    // provide a distinct final-use event on the retained stream (usually the
    // event recorded after target verification has consumed the prediction).
    [[nodiscard]] bool final_use(Ticket ticket, std::uintptr_t event,
                                 int error = 0) noexcept {
        if (!matches(ticket) || phase_ != Exl3MtpRunnerPhase::Ready || event == 0 ||
            event == producer_event_) {
            // A bad final-use witness cannot make the live output resettable.
            reason_ = Exl3MtpFallbackReason::EventMismatch;
            return false;
        }
        if (error != 0) {
            fail(Exl3MtpFallbackReason::ExecutionFailed);
            return false;
        }
        final_use_event_ = event;
        phase_ = Exl3MtpRunnerPhase::Retired;
        reason_ = Exl3MtpFallbackReason::None;
        return true;
    }

    // Drop an unsubmitted/failed/consumed step.  A submitted or ready step
    // cannot be reset without a witnessed final use; this prevents a DFlash2
    // repair/cancellation path from releasing a live MTP KV/output owner.
    [[nodiscard]] bool reset() noexcept {
        if (phase_ == Exl3MtpRunnerPhase::Submitted || phase_ == Exl3MtpRunnerPhase::Ready) {
            // Preserve the live phase so the caller can still supply the
            // required producer/final-use witness.  Turning this into Failed
            // would let a subsequent reset release asynchronous storage.
            reason_ = Exl3MtpFallbackReason::MissingFinalUse;
            return false;
        }
        input_            = {};
        output_           = {};
        output_owner_.reset();
        retained_stream_ = 0;
        producer_event_  = 0;
        final_use_event_ = 0;
        phase_            = Exl3MtpRunnerPhase::Idle;
        reason_           = Exl3MtpFallbackReason::None;
        return true;
    }

private:
    [[nodiscard]] bool matches(Ticket ticket) const noexcept {
        return ticket && ticket.generation == generation_ &&
               ticket.request_identity == input_.request_identity;
    }

    void clear_for_new_step() noexcept {
        input_            = {};
        output_           = {};
        output_owner_.reset();
        retained_stream_ = 0;
        producer_event_  = 0;
        final_use_event_ = 0;
        reason_           = Exl3MtpFallbackReason::None;
    }

    void fail(Exl3MtpFallbackReason reason) noexcept {
        reason_ = reason;
        phase_  = Exl3MtpRunnerPhase::Failed;
    }

    [[nodiscard]] Ticket fail_ticket(Exl3MtpFallbackReason reason) noexcept {
        fail(reason);
        return {};
    }

    Exl3MtpExecutionBinding binding_;
    std::shared_ptr<const void> mtp_kv_owner_;
    std::shared_ptr<const void> stream_owner_;
    Exl3MtpEnqueueOneStep enqueue_ = nullptr;
    void* user_                    = nullptr;

    Exl3MtpRunnerPhase phase_         = Exl3MtpRunnerPhase::Idle;
    Exl3MtpFallbackReason reason_     = Exl3MtpFallbackReason::None;
    std::uint64_t generation_         = 0;
    std::uintptr_t retained_stream_   = 0;
    std::uintptr_t producer_event_    = 0;
    std::uintptr_t final_use_event_   = 0;
    Exl3MtpStepInput input_;
    Exl3MtpStepOutput output_;
    std::shared_ptr<const void> output_owner_;
};

// Smallest real native EXL3 MTP numerical executor. It owns all intermediate
// activations and K4/K6 workspaces, consumes the strict 39-tensor binding plus
// the target's actual embedding/H6 head, and submits exactly one bridge step.
// The executor is deliberately not a scheduler and has no production caller
// by default; callers install its callback only behind an explicit experiment
// flag and keep target verification authoritative.
class Exl3NativeMtpOneStepExecutor {
public:
    explicit Exl3NativeMtpOneStepExecutor(Exl3MtpExecutionBinding binding);
    ~Exl3NativeMtpOneStepExecutor();

    Exl3NativeMtpOneStepExecutor(const Exl3NativeMtpOneStepExecutor&) = delete;
    Exl3NativeMtpOneStepExecutor& operator=(const Exl3NativeMtpOneStepExecutor&) = delete;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const Exl3MtpExecutionBinding& binding() const noexcept;

    // Prefix population is serialized on one retained stream.  This updates
    // only the executor's expected chronological KV frontier; it does not
    // publish a row or mutate target authority.  The caller records the
    // producer event after the corresponding enqueue and owns the fallback
    // decision if the next row is rejected.
    [[nodiscard]] bool set_mtp_kv_valid_rows(std::uint32_t rows) noexcept;

    // Returns false before any enqueue when the binding/shape/KV contract is
    // not admitted. Exceptions from CUDA submission are converted to false so
    // the owner falls back to ordinary target verification.
    [[nodiscard]] bool enqueue(const Exl3MtpStepInput& input,
                               const Exl3MtpStepOutput& output,
                               std::uintptr_t retained_stream) noexcept;

    static bool enqueue_callback(const Exl3MtpExecutionBinding& binding,
                                 const Exl3MtpStepInput& input,
                                 const Exl3MtpStepOutput& output,
                                 std::uintptr_t retained_stream,
                                 void* user) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Owned default-off bridge for the first real DFlash2 -> native-MTP path.
// It loads exactly the closed 39-tensor manifest, allocates separate MTP KV,
// and teacher-forces a bounded sequence of hidden[p]+token[p+1] pairs through
// the one-step executor.  The class deliberately has no target-state commit
// operation: callers retain DFlash2/target authority and use the returned
// event only as a readiness witness for the proposal output.
class Exl3MtpPrefixState {
public:
    static std::shared_ptr<Exl3MtpPrefixState> materialize(
        const IndexedSafetensors& collection,
        Exl3MtpSharedTargetBinding shared_target,
        std::uint64_t model_identity,
        std::uint32_t kv_capacity);

    ~Exl3MtpPrefixState();
    Exl3MtpPrefixState(const Exl3MtpPrefixState&) = delete;
    Exl3MtpPrefixState& operator=(const Exl3MtpPrefixState&) = delete;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const Exl3MtpExecutionBinding& binding() const noexcept;
    [[nodiscard]] std::shared_ptr<const void> owner() const noexcept;
    [[nodiscard]] Exl3MtpBuffer last_hidden() const noexcept;
    [[nodiscard]] Exl3MtpBuffer last_logits() const noexcept;
    [[nodiscard]] Exl3MtpBuffer last_predicted_token() const noexcept;

    // Populate rows in strict chronological order.  `hidden_rows[i]` must be
    // an owned final-normalized target row at position
    // binding.mtp_kv.valid_rows+i.  `token_ids` is the contiguous sequence
    // window beginning at that same position, so token_ids[0] is the token at
    // the first hidden row and token_ids[i+1] is the exact token consumed by
    // row i.  The extra trailing token is intentional: it may be the first
    // DFlash2 suffix token used to produce the native MTP continuation.
    [[nodiscard]] Exl3MtpPrefixReceipt prefill_target_prefix(
        std::span<const Exl3MtpTargetHiddenHandoff> hidden_rows,
        std::span<const std::int64_t> token_ids,
        std::uint64_t request_identity,
        std::uint64_t target_identity,
        std::uintptr_t retained_stream) noexcept;

    // Consume one supplied suffix token from the final MTP hidden row and
    // produce the next native-MTP proposal.  This is the smallest direct
    // DFlash2 -> MTP extension: target verification remains responsible for
    // deciding whether the returned token is committed.
    [[nodiscard]] Exl3MtpPrefixReceipt ar_step(
        std::int64_t token, std::uint64_t request_identity,
        std::uint64_t target_identity, std::uintptr_t retained_stream) noexcept;

    // Wait for the last producer event, release retained handoff owners, and
    // return to an empty KV frontier.  A failed/poisoned population is only
    // reusable after this reset succeeds.
    [[nodiscard]] bool reset() noexcept;

private:
    struct Impl;
    explicit Exl3MtpPrefixState(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::exl3
