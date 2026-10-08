#pragma once
// Batched multi-agent rounds: lanes of the coherent-device Engine meet at the
// target verification of their DFlash2 rounds. The first lane to arrive opens
// a group and waits briefly; later arrivals join it. The group closes when
// every active lane of this coordinator has arrived (at most four, the widest
// batched forward) or when the opener's wait ends. The closing thread runs
// one batched target forward (Exl3TextContext::continue_rows_batched) led by
// the member registered first, on that member's stream after every member's
// queued work; each member resumes on its own stream after the forward. A
// lane verifies alone when it is the only active lane or nobody joins in time.
#include "exl3/text_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::exl3 {

class Exl3BatchedVerifyCoordinator {
public:
    static constexpr std::size_t kMaxGroup = Exl3TextContext::kMaxBatchedPeers + 1;

    explicit Exl3BatchedVerifyCoordinator(std::chrono::microseconds wait) : wait_(wait) {}
    ~Exl3BatchedVerifyCoordinator() {
        for (auto& [context, lane] : lanes_) {
            (void)context;
            cudaEventDestroy(lane.ready);
            cudaEventDestroy(lane.done);
        }
    }
    Exl3BatchedVerifyCoordinator(const Exl3BatchedVerifyCoordinator&) = delete;
    Exl3BatchedVerifyCoordinator& operator=(const Exl3BatchedVerifyCoordinator&) = delete;

    // Lanes register in a fixed order; the earliest registered member of a
    // group leads its forward (captured graphs are prepared per leader and
    // ordered peer list).
    void register_context(Exl3TextContext& context) {
        std::lock_guard lock(mutex_);
        lane_for(context);
    }

    // Every group of two or more registered lanes (up to kMaxGroup), as the
    // ordered member lists a batched forward uses (leader first).
    std::vector<std::vector<Exl3TextContext*>> groups() const {
        std::lock_guard lock(mutex_);
        std::vector<Exl3TextContext*> ordered(lanes_.size());
        for (const auto& [context, lane] : lanes_) ordered[lane.order] = const_cast<Exl3TextContext*>(context);
        std::vector<std::vector<Exl3TextContext*>> result;
        const std::size_t n = ordered.size();
        for (std::uint32_t mask = 1; mask < (1u << n); ++mask) {
            std::vector<Exl3TextContext*> members;
            for (std::size_t i = 0; i < n; ++i)
                if (mask & (1u << i)) members.push_back(ordered[i]);
            if (members.size() >= 2 && members.size() <= kMaxGroup) result.push_back(std::move(members));
        }
        return result;
    }

    // A lane holds an active request (decode rounds may arrive).
    void set_active(bool active) {
        std::lock_guard lock(mutex_);
        active_ += active ? 1 : -1;
        changed_.notify_all();
    }

    void continue_rows(Exl3TextContext& context, std::span<const std::int64_t> tokens,
                       cudaStream_t stream) {
        std::unique_lock lock(mutex_);
        auto& lane = lane_for(context);
        if (active_ < 2 && !open_) {
            ++solo_;
            lock.unlock();
            context.continue_rows(tokens, stream);
            return;
        }
        Member mine;
        mine.context = &context;
        mine.order = lane.order;
        mine.stream = stream;
        mine.tokens.assign(tokens.begin(), tokens.end());
        mine.ready = lane.ready;
        mine.done = lane.done;
        check(cudaEventRecord(mine.ready, stream), "batched verify arrival");
        const auto arrived = std::chrono::steady_clock::now();
        std::shared_ptr<Group> group;
        bool opener = false;
        if (open_) {
            group = open_;
        } else {
            group = std::make_shared<Group>();
            open_ = group;
            opener = true;
        }
        group->members.push_back(&mine);
        const auto target = std::min<std::size_t>(max_group(), static_cast<std::size_t>(std::max(active_, 1)));
        if (group->members.size() >= target) {
            open_ = nullptr;
            lead(lock, *group);
        } else if (opener) {
            const auto deadline = arrived + wait_;
            while (!group->closed && changed_.wait_until(lock, deadline) != std::cv_status::timeout) {
                // Fewer active lanes now: close once everyone left has arrived.
                const auto now_target = std::min<std::size_t>(max_group(),
                    static_cast<std::size_t>(std::max(active_, 1)));
                if (!group->closed && group->members.size() >= now_target) break;
            }
            if (!group->closed) {
                if (open_ == group) open_ = nullptr;
                if (group->members.size() < 2) ++timeouts_;
                lead(lock, *group);
            }
        }
        changed_.wait(lock, [&] { return group->done; });
        wait_us_ += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - arrived).count());
        const auto error = group->error;
        lock.unlock();
        if (error) std::rethrow_exception(error);
    }

    std::uint64_t batched_rounds() const {
        std::lock_guard lock(mutex_);
        return batched_;
    }
    std::uint64_t solo_rounds() const {
        std::lock_guard lock(mutex_);
        return solo_;
    }
    std::uint64_t eager_rounds() const {
        std::lock_guard lock(mutex_);
        return eager_;
    }
    std::uint64_t timeouts() const {
        std::lock_guard lock(mutex_);
        return timeouts_;
    }
    std::uint64_t wait_us() const {
        std::lock_guard lock(mutex_);
        return wait_us_;
    }
    // Batched rounds by group width (index = members).
    std::array<std::uint64_t, kMaxGroup + 1> width_histogram() const {
        std::lock_guard lock(mutex_);
        return widths_;
    }

private:
    struct Lane {
        std::size_t order = 0;
        cudaEvent_t ready = nullptr;
        cudaEvent_t done = nullptr;
    };
    struct Member {
        Exl3TextContext* context = nullptr;
        std::size_t order = 0;
        std::vector<std::int64_t> tokens;
        cudaStream_t stream = nullptr;
        cudaEvent_t ready = nullptr;
        cudaEvent_t done = nullptr;
    };
    struct Group {
        std::vector<Member*> members;
        bool closed = false;
        bool done = false;
        std::exception_ptr error;
    };
    // NINFER_EXL3_BATCHED_MAX_GROUP (2..4): widest group (research).
    static std::size_t max_group() {
        static const std::size_t value = [] {
            const char* text = std::getenv("NINFER_EXL3_BATCHED_MAX_GROUP");
            const int parsed = text ? std::atoi(text) : static_cast<int>(kMaxGroup);
            return static_cast<std::size_t>(std::max(2, std::min(static_cast<int>(kMaxGroup), parsed)));
        }();
        return value;
    }
    static void check(cudaError_t error, const char* what) {
        if (error != cudaSuccess)
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }
    Lane& lane_for(Exl3TextContext& context) {
        auto [it, inserted] = lanes_.try_emplace(&context);
        if (inserted) {
            it->second.order = lanes_.size() - 1;
            check(cudaEventCreateWithFlags(&it->second.ready, cudaEventDisableTiming), "batched verify event");
            check(cudaEventCreateWithFlags(&it->second.done, cudaEventDisableTiming), "batched verify event");
        }
        return it->second;
    }
    // Runs the group's forward with the lock released; every member then
    // resumes after group.done_event (the leader's completion event).
    void lead(std::unique_lock<std::mutex>& lock, Group& group) {
        group.closed = true;
        std::sort(group.members.begin(), group.members.end(),
                  [](const Member* a, const Member* b) { return a->order < b->order; });
        auto members = group.members;
        lock.unlock();
        std::exception_ptr error;
        bool eager = false;
        try {
            if (members.size() == 1) {
                members[0]->context->continue_rows(members[0]->tokens, members[0]->stream);
            } else {
                Member& leader = *members[0];
                std::vector<Exl3TextContext::BatchedPeer> peers;
                for (std::size_t i = 1; i < members.size(); ++i) {
                    check(cudaStreamWaitEvent(leader.stream, members[i]->ready, 0), "batched verify peer order");
                    peers.push_back({members[i]->context, members[i]->tokens, members[i]->stream});
                }
                const auto graphs_before = leader.context->batched_graph_replays();
                leader.context->continue_rows_batched(peers, leader.tokens, leader.stream);
                check(cudaEventRecord(leader.done, leader.stream), "batched verify completion");
                // Members resume after the forward; ordered here, before the
                // leader can record its event again for a later round.
                for (std::size_t i = 1; i < members.size(); ++i)
                    check(cudaStreamWaitEvent(members[i]->stream, leader.done, 0), "batched verify resume");
                eager = leader.context->batched_graph_replays() == graphs_before;
            }
        } catch (...) {
            error = std::current_exception();
        }
        lock.lock();
        if (members.size() == 1) ++solo_;
        else {
            ++batched_;
            ++widths_[members.size()];
            eager_ += eager ? 1 : 0;
        }
        group.error = error;
        group.done = true;
        changed_.notify_all();
    }

    const std::chrono::microseconds wait_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::map<const Exl3TextContext*, Lane> lanes_;
    std::shared_ptr<Group> open_;
    int active_ = 0;
    std::uint64_t batched_ = 0, solo_ = 0, timeouts_ = 0, wait_us_ = 0, eager_ = 0;
    std::array<std::uint64_t, kMaxGroup + 1> widths_{};
};

}  // namespace ninfer::exl3
