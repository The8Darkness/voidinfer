#pragma once

#include "exl3/mtp_runner.h"

#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::exl3 {

// The campaign uses the same two 4K/128 token fixtures as the selected DFlash2
// screen.  The catalog is deliberately path-injected: this header never
// invents a fixture, reads a model artifact, or silently substitutes a prompt.
enum class Exl3MtpFixtureKind : std::uint8_t { Code, Prose };

struct Exl3MtpFixtureIdentity {
    Exl3MtpFixtureKind kind = Exl3MtpFixtureKind::Code;
    std::string label;
    std::filesystem::path ids_path;
    std::uint32_t prompt_tokens = 4'096;
    std::uint32_t output_tokens = 128;

    [[nodiscard]] bool valid() const noexcept {
        return !label.empty() && !ids_path.empty() && prompt_tokens != 0 && output_tokens != 0;
    }

    [[nodiscard]] std::string_view kind_name() const noexcept {
        return kind == Exl3MtpFixtureKind::Code ? "code" : "prose";
    }

    // NINFER_COMPACT_PERF_DIR is the existing source-of-truth fixture
    // directory used by the DFlash2 4K/128 screens.  No existence check is
    // performed here; the owned benchmark harness performs that check before
    // loading ids and records a missing fixture as NOT_RUN.
    [[nodiscard]] static Exl3MtpFixtureIdentity code(const std::filesystem::path& directory) {
        return {Exl3MtpFixtureKind::Code, "code", directory / "code.ids", 4'096, 128};
    }

    [[nodiscard]] static Exl3MtpFixtureIdentity prose(const std::filesystem::path& directory) {
        return {Exl3MtpFixtureKind::Prose, "prose", directory / "prose.ids", 4'096, 128};
    }
};

// Timings are supplied by the real numeric executor and target oracle.  An
// absent value is intentional: zero is never used for work that was not
// measured.  preparation_ms includes only target-hidden preparation required
// by this MTP proposal, not the ordinary DFlash2 prompt path.
struct Exl3MtpProposalTiming {
    std::optional<double> target_hidden_preparation_ms;
    std::optional<double> projection_ms;
    std::optional<double> module_ms;
    std::optional<double> token_selection_ms;
    std::optional<double> target_verification_ms;
    std::optional<double> repair_ms;
    std::optional<std::size_t> additional_workspace_bytes;

    [[nodiscard]] std::optional<double> proposal_cost_ms() const noexcept {
        if (!target_hidden_preparation_ms || !projection_ms || !module_ms ||
            !token_selection_ms) {
            return std::nullopt;
        }
        return *target_hidden_preparation_ms + *projection_ms + *module_ms + *token_selection_ms;
    }

    [[nodiscard]] std::optional<double> complete_round_ms() const noexcept {
        const auto proposal = proposal_cost_ms();
        if (!proposal || !target_verification_ms || !repair_ms) return std::nullopt;
        return *proposal + *target_verification_ms + *repair_ms;
    }
};

// Observation filled by an actual one-step executor.  The callback must not
// mutate HostKV/target authority; it may only produce the MTP output and its
// measurement.  The target oracle is a separate callback so a fast MTP
// module cannot be mistaken for a useful speculative proposal.
struct Exl3MtpProposalObservation {
    std::int32_t chosen_token = -1;
    Exl3MtpProposalTiming timing;
    std::uint64_t target_rows_evaluated = 0;
    std::uint64_t verifier_invocations = 0;
    std::uint64_t repair_rows = 0;
    bool target_authority_preserved = false;
};

using Exl3MtpProposalExecute = bool (*)(const Exl3MtpExecutionBinding& binding,
                                        const Exl3MtpStepInput& input,
                                        const Exl3MtpStepOutput& output,
                                        Exl3MtpProposalObservation& observation,
                                        void* user) noexcept;

// The oracle must use the same target/HostKV authority as the matched DFlash2
// screen.  It returns the target's actual greedy token for input.position+1,
// plus the complete verification/repair accounting for this one-step round.
using Exl3MtpTargetGreedyOracle = bool (*)(const Exl3MtpFixtureIdentity& fixture,
                                           std::int64_t position,
                                           std::int32_t& target_token,
                                           Exl3MtpProposalObservation& accounting,
                                           void* user) noexcept;

enum class Exl3MtpEconomicsStatus : std::uint8_t {
    NotRunMissingFixture,
    NotRunMissingExecutor,
    NotRunMissingTargetOracle,
    NotRunExecutorRejected,
    NotRunAuthorityViolation,
    Measured,
};

struct Exl3MtpProposalEconomicsResult {
    Exl3MtpFixtureIdentity fixture;
    Exl3MtpEconomicsStatus status = Exl3MtpEconomicsStatus::NotRunMissingExecutor;
    std::string status_detail;
    std::int64_t position = -1;
    std::optional<std::int32_t> mtp_token;
    std::optional<std::int32_t> target_greedy_token;
    std::optional<bool> position_one_match;
    Exl3MtpProposalTiming timing;
    std::uint64_t mtp_proposal_rows = 1;
    std::uint64_t target_rows_evaluated = 0;
    std::uint64_t verifier_invocations = 0;
    std::uint64_t repair_rows = 0;
    std::uint64_t useful_committed_tokens = 0;
    std::optional<double> expected_useful_tokens;
    std::optional<double> useful_commits_per_target_row;
    std::optional<double> useful_commits_per_verifier_invocation;
    std::optional<double> complete_ms_per_useful_token;

    [[nodiscard]] bool measured() const noexcept {
        return status == Exl3MtpEconomicsStatus::Measured;
    }

    [[nodiscard]] std::string_view status_name() const noexcept {
        switch (status) {
        case Exl3MtpEconomicsStatus::NotRunMissingFixture:
            return "NOT_RUN_MISSING_FIXTURE";
        case Exl3MtpEconomicsStatus::NotRunMissingExecutor:
            return "NOT_RUN_MISSING_EXECUTOR";
        case Exl3MtpEconomicsStatus::NotRunMissingTargetOracle:
            return "NOT_RUN_MISSING_TARGET_ORACLE";
        case Exl3MtpEconomicsStatus::NotRunExecutorRejected:
            return "NOT_RUN_EXECUTOR_REJECTED";
        case Exl3MtpEconomicsStatus::NotRunAuthorityViolation:
            return "NOT_RUN_AUTHORITY_VIOLATION";
        case Exl3MtpEconomicsStatus::Measured:
            return "MEASURED";
        }
        return "NOT_RUN_UNKNOWN";
    }
};

// This is intentionally a one-step screen.  It is an adapter seam for the
// real TextContext::mtp_forward_ar_step/EXL3 executor, not a second scheduler.
// No callback means no arithmetic is attempted and the result remains an
// explicit fail-closed record.  The caller owns the hidden/KV/output/event
// lifetimes and must keep them valid for the duration of the callback.
class Exl3MtpProposalEconomicsScreen {
public:
    [[nodiscard]] static Exl3MtpProposalEconomicsResult measure(
        Exl3MtpFixtureIdentity fixture, const Exl3MtpExecutionBinding& binding,
        const Exl3MtpStepInput& input, const Exl3MtpStepOutput& output,
        Exl3MtpProposalExecute execute, Exl3MtpTargetGreedyOracle target_oracle,
        void* user = nullptr) noexcept {
        Exl3MtpProposalEconomicsResult result;
        result.fixture = std::move(fixture);
        result.position = input.position;
        if (!result.fixture.valid()) {
            result.status = Exl3MtpEconomicsStatus::NotRunMissingFixture;
            result.status_detail = "fixture identity/path/extent is incomplete";
            return result;
        }
        if (execute == nullptr) {
            result.status = Exl3MtpEconomicsStatus::NotRunMissingExecutor;
            result.status_detail = "numeric MTP executor callback is not installed";
            return result;
        }
        if (target_oracle == nullptr) {
            result.status = Exl3MtpEconomicsStatus::NotRunMissingTargetOracle;
            result.status_detail = "target greedy/HostKV oracle callback is not installed";
            return result;
        }
        if (!binding.valid() || !input.valid() ||
            !output.hidden.exact(kExl3MtpHiddenWidth, 1,
                                 static_cast<std::size_t>(kExl3MtpHiddenWidth) * 2U) ||
            !output.logits.exact(kExl3MtpVocabSize, 1,
                                 static_cast<std::size_t>(kExl3MtpVocabSize) * 2U) ||
            !output.predicted_token.exact(1, 1, sizeof(std::int32_t))) {
            result.status = Exl3MtpEconomicsStatus::NotRunExecutorRejected;
            result.status_detail = "one-step input/output contract is invalid";
            return result;
        }

        Exl3MtpProposalObservation proposal;
        if (!execute(binding, input, output, proposal, user)) {
            result.status = Exl3MtpEconomicsStatus::NotRunExecutorRejected;
            result.status_detail = "numeric executor rejected the one-step proposal";
            return result;
        }
        if (!proposal.target_authority_preserved) {
            result.status = Exl3MtpEconomicsStatus::NotRunAuthorityViolation;
            result.status_detail = "proposal executor did not preserve target authority";
            return result;
        }

        result.mtp_token = proposal.chosen_token;
        result.timing = proposal.timing;
        result.target_rows_evaluated = proposal.target_rows_evaluated;
        result.verifier_invocations = proposal.verifier_invocations;
        result.repair_rows = proposal.repair_rows;

        std::int32_t target_token = -1;
        Exl3MtpProposalObservation accounting;
        if (!target_oracle(result.fixture, input.position, target_token, accounting, user)) {
            result.status = Exl3MtpEconomicsStatus::NotRunMissingTargetOracle;
            result.status_detail = "target oracle did not produce an authoritative token";
            return result;
        }
        result.target_greedy_token = target_token;
        result.timing.target_verification_ms = accounting.timing.target_verification_ms;
        result.timing.repair_ms = accounting.timing.repair_ms;
        result.target_rows_evaluated = accounting.target_rows_evaluated;
        result.verifier_invocations = accounting.verifier_invocations;
        result.repair_rows = accounting.repair_rows;
        if (!accounting.target_authority_preserved || target_token < 0 || proposal.chosen_token < 0) {
            result.status = Exl3MtpEconomicsStatus::NotRunAuthorityViolation;
            result.status_detail = "target oracle did not attest target authority/token";
            return result;
        }

        result.position_one_match = proposal.chosen_token == target_token;
        result.useful_committed_tokens = *result.position_one_match ? 1 : 0;
        result.expected_useful_tokens = result.position_one_match ? std::optional<double>{1.0}
                                                                  : std::optional<double>{0.0};
        if (result.target_rows_evaluated != 0) {
            result.useful_commits_per_target_row =
                static_cast<double>(result.useful_committed_tokens) /
                static_cast<double>(result.target_rows_evaluated);
        }
        if (result.verifier_invocations != 0) {
            result.useful_commits_per_verifier_invocation =
                static_cast<double>(result.useful_committed_tokens) /
                static_cast<double>(result.verifier_invocations);
        }
        if (const auto complete = result.timing.complete_round_ms();
            complete && result.useful_committed_tokens != 0) {
            result.complete_ms_per_useful_token = *complete;
        }
        result.status = Exl3MtpEconomicsStatus::Measured;
        result.status_detail = "one-step fixture measurement; aggregate before policy change";
        return result;
    }
};

} // namespace ninfer::exl3
