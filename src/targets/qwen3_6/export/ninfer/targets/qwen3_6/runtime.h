#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::targets::qwen3_6 {

enum class TextPhase {
    Prefill,
    Verify,
};

struct GraphExecutionProfile {
    std::uint32_t min            = 0;
    std::uint32_t max            = 0;
    std::uint32_t topology_class = 0;
};

namespace detail {
template <class Variant>
struct SequencePlanImpl;
template <class Variant>
struct SequencePlannerImpl;
template <class Variant>
struct AdmissionPlanImpl;
template <class Variant>
struct RequestBasePlanImpl;
template <class Variant>
class ProgramImpl;
template <class Variant>
struct RuntimeContractAccess;
} // namespace detail

template <class Variant>
class SequencePlanner;

// These are the complete family execution types. Exact packages bind them to a private Variant;
// target selection remains outside this layer and happens once in the closed Engine registry.
//
// The plan types declare their moves/move-assignments here and define them per variant in
// api_impl.h with explicit bodies: any implicit definition needs the complete detail impl types
// (only the exact target TUs have them), and MSVC 19.44 does not emit out-of-line `= default`
// explicit specializations of these moves (LNK2019 at the final Windows link).
template <class Variant>
class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t request_transient_capacity_bytes() const noexcept;

public:
    // Family-private construction/storage seam; exact packages expose only the completed alias.
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl<Variant>> impl_;

    template <class V>
    friend class SequencePlanner;
    template <class V>
    friend class detail::ProgramImpl;
};

template <class Variant>
class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept;
    [[nodiscard]] SequencePlan<Variant> finalize(std::uint32_t main_page_groups) &&;

public:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl_;

    template <class V>
    friend SequencePlanner<V> make_sequence_planner(DeviceContext&, const EngineOptions&,
                                                    typename V::WeightsProfile);
};

template <class Variant>
class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

public:
    explicit RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl_;
};

template <class Variant>
class AdmissionPlan {
public:
    AdmissionPlan(AdmissionPlan&&) noexcept;
    AdmissionPlan& operator=(AdmissionPlan&&) noexcept;
    ~AdmissionPlan();

    AdmissionPlan(const AdmissionPlan&)            = delete;
    AdmissionPlan& operator=(const AdmissionPlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

public:
    // Family-private construction/storage seam. Exact packages expose only the completed alias;
    // Engine code can inspect summary() but not target planning state.
    explicit AdmissionPlan(std::unique_ptr<detail::AdmissionPlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::AdmissionPlanImpl<Variant>> impl_;
};

template <class Variant>
class SequenceHandle {
public:
    SequenceHandle() noexcept                                 = default;
    SequenceHandle(const SequenceHandle&) noexcept            = default;
    SequenceHandle& operator=(const SequenceHandle&) noexcept = default;

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;

    friend struct detail::RuntimeContractAccess<Variant>;
};

template <class Variant>
class ContinuationHandle {
public:
    ContinuationHandle() noexcept = default;
    ~ContinuationHandle()         = default;

    ContinuationHandle(ContinuationHandle&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), lane_(other.lane_),
          epoch_(std::exchange(other.epoch_, 0)) {}

    ContinuationHandle& operator=(ContinuationHandle&&)      = delete;
    ContinuationHandle(const ContinuationHandle&)            = delete;
    ContinuationHandle& operator=(const ContinuationHandle&) = delete;

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;

    friend struct detail::RuntimeContractAccess<Variant>;
};

template <class Variant>
class PendingBatch {
public:
    PendingBatch() noexcept = default;
    ~PendingBatch()         = default;

    PendingBatch(PendingBatch&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          transaction_(std::exchange(other.transaction_, 0)), rows_(other.rows_),
          row_count_(std::exchange(other.row_count_, 0)), tokens_(other.tokens_),
          row_counts_(other.row_counts_), row_stride_(other.row_stride_) {
        other.tokens_     = {};
        other.row_counts_ = {};
        other.row_stride_ = 0;
    }

    PendingBatch& operator=(PendingBatch&&)      = delete;
    PendingBatch(const PendingBatch&)            = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }

    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }

    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return row_counts_; }

    [[nodiscard]] std::uint32_t row_stride() const noexcept { return row_stride_; }

private:
    const void* owner_         = nullptr;
    std::uint64_t transaction_ = 0;
    std::array<SequenceHandle<Variant>, kMaximumConcurrency> rows_{};
    std::size_t row_count_ = 0;
    std::span<const TokenId> tokens_;
    std::span<const std::int32_t> row_counts_;
    std::uint32_t row_stride_ = 0;

    friend struct detail::RuntimeContractAccess<Variant>;
};

template <class Variant>
struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    std::optional<PendingBatch<Variant>> pending;
};

template <class Variant>
struct StartResult {
    SequenceHandle<Variant> sequence;
    runtime::AdmissionResources active_resources;
    PrefillProgress<Variant> progress;
};

struct CommitRowResult {
    runtime::CommitDisposition disposition = runtime::CommitDisposition::Active;
    runtime::AdmissionResources released_resources;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

template <class Variant>
struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    std::size_t row_count = 0;
};

template <class Variant>
struct DiscardResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    std::array<runtime::AdmissionResources, kMaximumConcurrency> released_resources{};
    std::size_t row_count = 0;
};

template <class Variant>
struct FinishResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
    runtime::AdmissionResources released_resources;
    runtime::AdmissionResources resident_resources;
    std::optional<ContinuationHandle<Variant>> continuation;
};

template <class Variant>
struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
    runtime::AdmissionResources released_resources;
};

template <class Variant>
struct ReleaseResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    runtime::AdmissionResources released_resources;
};

template <class Variant>
class Program {
public:
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // Engine owns scheduling and logical residency policy. Program owns physical lanes, opaque
    // capabilities, model state and one immutable pending transaction at a time.
    [[nodiscard]] RequestBasePlan<Variant>
    plan_request(const PreparedPrompt& prompt, const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] AdmissionPlan<Variant>
    inspect_admission(const PreparedPrompt& prompt, const RequestBasePlan<Variant>& base,
                      runtime::LaneId destination, const ContinuationHandle<Variant>* source);
    [[nodiscard]] StartResult<Variant>
    start_request(AdmissionPlan<Variant>&& plan, PreparedPrompt&& prompt,
                  std::optional<ContinuationHandle<Variant>>&& source);
    [[nodiscard]] PrefillProgress<Variant> advance_prefill(SequenceHandle<Variant> sequence);
    [[nodiscard]] PendingBatch<Variant> decode(std::span<const SequenceHandle<Variant>> sequences,
                                               std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] CommitResult<Variant>
    commit(PendingBatch<Variant>&& pending, std::span<const runtime::CommitDecision> decisions,
           runtime::CommitObservation observation = runtime::CommitObservation::AllRows);
    [[nodiscard]] DiscardResult<Variant> abort_pending(PendingBatch<Variant>&& pending) noexcept;
    [[nodiscard]] FinishResult<Variant> finish(SequenceHandle<Variant> sequence,
                                               runtime::RetentionDecision decision) noexcept;
    [[nodiscard]] AbortResult<Variant> abort(SequenceHandle<Variant> sequence) noexcept;
    [[nodiscard]] ReleaseResult<Variant>
    release_continuation(ContinuationHandle<Variant>&& continuation) noexcept;
    void fail_all_cleanup() noexcept;

    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

private:
    explicit Program(std::unique_ptr<detail::ProgramImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl<Variant>> impl_;

    template <class V>
    friend std::unique_ptr<Program<V>> create_program(const typename V::ModelView&,
                                                      typename V::WeightsProfile, SequencePlan<V>&&,
                                                      DeviceContext&);
};

namespace detail {

template <class Variant>
struct RuntimeContractAccess {
    [[nodiscard]] static SequenceHandle<Variant>
    make_sequence(const void* owner, runtime::LaneId lane, std::uint64_t epoch) noexcept {
        SequenceHandle<Variant> out;
        out.owner_ = owner;
        out.lane_  = lane;
        out.epoch_ = epoch;
        return out;
    }

    [[nodiscard]] static ContinuationHandle<Variant>
    make_continuation(const void* owner, runtime::LaneId lane, std::uint64_t epoch) noexcept {
        ContinuationHandle<Variant> out;
        out.owner_ = owner;
        out.lane_  = lane;
        out.epoch_ = epoch;
        return out;
    }

    [[nodiscard]] static const void* owner(const SequenceHandle<Variant>& handle) noexcept {
        return handle.owner_;
    }

    [[nodiscard]] static runtime::LaneId lane(const SequenceHandle<Variant>& handle) noexcept {
        return handle.lane_;
    }

    [[nodiscard]] static std::uint64_t epoch(const SequenceHandle<Variant>& handle) noexcept {
        return handle.epoch_;
    }

    [[nodiscard]] static const void* owner(const ContinuationHandle<Variant>& handle) noexcept {
        return handle.owner_;
    }

    [[nodiscard]] static runtime::LaneId lane(const ContinuationHandle<Variant>& handle) noexcept {
        return handle.lane_;
    }

    [[nodiscard]] static std::uint64_t epoch(const ContinuationHandle<Variant>& handle) noexcept {
        return handle.epoch_;
    }

    static void consume(ContinuationHandle<Variant>& handle) noexcept {
        handle.owner_ = nullptr;
        handle.epoch_ = 0;
    }

    [[nodiscard]] static PendingBatch<Variant>
    make_pending(const void* owner, std::uint64_t transaction,
                 std::span<const SequenceHandle<Variant>> rows, std::span<const TokenId> tokens,
                 std::span<const std::int32_t> row_counts, std::uint32_t row_stride) {
        PendingBatch<Variant> out;
        out.owner_       = owner;
        out.transaction_ = transaction;
        out.row_count_   = rows.size();
        for (std::size_t i = 0; i < rows.size(); ++i) { out.rows_[i] = rows[i]; }
        out.tokens_     = tokens;
        out.row_counts_ = row_counts;
        out.row_stride_ = row_stride;
        return out;
    }

    [[nodiscard]] static const void* owner(const PendingBatch<Variant>& pending) noexcept {
        return pending.owner_;
    }

    [[nodiscard]] static std::uint64_t transaction(const PendingBatch<Variant>& pending) noexcept {
        return pending.transaction_;
    }

    [[nodiscard]] static std::span<const SequenceHandle<Variant>>
    rows(const PendingBatch<Variant>& pending) noexcept {
        return {pending.rows_.data(), pending.row_count_};
    }

    static void consume(PendingBatch<Variant>& pending) noexcept {
        pending.owner_       = nullptr;
        pending.transaction_ = 0;
        pending.row_count_   = 0;
        pending.tokens_      = {};
        pending.row_counts_  = {};
        pending.row_stride_  = 0;
    }
};

} // namespace detail

template <class Variant>
[[nodiscard]] SequencePlanner<Variant>
make_sequence_planner(DeviceContext& device, const EngineOptions& options,
                      typename Variant::WeightsProfile weights_profile);

template <class Variant>
[[nodiscard]] std::unique_ptr<Program<Variant>>
create_program(const typename Variant::ModelView& model,
               typename Variant::WeightsProfile weights_profile, SequencePlan<Variant>&& plan,
               DeviceContext& device);

} // namespace ninfer::targets::qwen3_6
