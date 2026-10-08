#pragma once
// Batched multi-agent rounds: lanes of the coherent-device Engine meet at the
// target verification of their DFlash2 rounds. The first lane to arrive waits
// briefly; when another active lane arrives, that thread runs one batched
// target forward for both (Exl3TextContext::continue_rows_batched) on its own
// stream after the waiting lane's work, and the waiting lane resumes on its
// own stream after the forward. A lane verifies alone when it is the only
// active lane or nobody arrives in time.
#include "exl3/text_model.h"

#include <cuda_runtime.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <map>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::exl3 {

class Exl3BatchedVerifyCoordinator {
public:
    explicit Exl3BatchedVerifyCoordinator(std::chrono::microseconds wait) : wait_(wait) {}
    ~Exl3BatchedVerifyCoordinator() {
        for (auto& [context, events] : events_) {
            (void)context;
            cudaEventDestroy(events.ready);
            cudaEventDestroy(events.done);
        }
    }
    Exl3BatchedVerifyCoordinator(const Exl3BatchedVerifyCoordinator&) = delete;
    Exl3BatchedVerifyCoordinator& operator=(const Exl3BatchedVerifyCoordinator&) = delete;

    // A lane holds an active request (decode rounds may arrive).
    void set_active(bool active) {
        std::lock_guard lock(mutex_);
        active_ += active ? 1 : -1;
        changed_.notify_all();
    }

    void continue_rows(Exl3TextContext& context, std::span<const std::int64_t> tokens,
                       cudaStream_t stream) {
        std::unique_lock lock(mutex_);
        auto& events = events_for(context);
        if (waiting_ && waiting_->context != &context) {
            Pending* peer = waiting_;
            waiting_ = nullptr;
            peer->taken = true;
            lock.unlock();
            std::exception_ptr error;
            try {
                check(cudaStreamWaitEvent(stream, peer->ready, 0), "batched verify peer order");
                const auto graphs_before = context.batched_graph_replays();
                context.continue_rows_batched(*peer->context, tokens, peer->tokens, stream,
                                              peer->stream);
                check(cudaEventRecord(peer->done, stream), "batched verify completion");
                eager_ += context.batched_graph_replays() == graphs_before ? 1 : 0;
            } catch (...) {
                error = std::current_exception();
            }
            lock.lock();
            peer->error = error;
            peer->done_flag = true;
            ++batched_;
            changed_.notify_all();
            lock.unlock();
            if (error) std::rethrow_exception(error);
            return;
        }
        if (active_ < 2 || waiting_) {
            ++solo_;
            lock.unlock();
            context.continue_rows(tokens, stream);
            return;
        }
        Pending mine;
        mine.context = &context;
        mine.stream = stream;
        mine.tokens.assign(tokens.begin(), tokens.end());
        mine.ready = events.ready;
        mine.done = events.done;
        check(cudaEventRecord(mine.ready, stream), "batched verify arrival");
        waiting_ = &mine;
        const auto arrived = std::chrono::steady_clock::now();
        const auto deadline = std::chrono::steady_clock::now() + wait_;
        while (!mine.taken && active_ >= 2 &&
               changed_.wait_until(lock, deadline) != std::cv_status::timeout) {}
        wait_us_ += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - arrived).count());
        if (!mine.taken) {
            if (waiting_ == &mine) waiting_ = nullptr;
            ++solo_;
            ++timeouts_;
            lock.unlock();
            context.continue_rows(tokens, stream);
            return;
        }
        changed_.wait(lock, [&] { return mine.done_flag; });
        lock.unlock();
        if (mine.error) std::rethrow_exception(mine.error);
        check(cudaStreamWaitEvent(stream, mine.done, 0), "batched verify resume");
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

private:
    struct Events {
        cudaEvent_t ready = nullptr;
        cudaEvent_t done = nullptr;
    };
    struct Pending {
        Exl3TextContext* context = nullptr;
        std::vector<std::int64_t> tokens;
        cudaStream_t stream = nullptr;
        cudaEvent_t ready = nullptr;
        cudaEvent_t done = nullptr;
        bool taken = false;
        bool done_flag = false;
        std::exception_ptr error;
    };
    static void check(cudaError_t error, const char* what) {
        if (error != cudaSuccess)
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }
    Events& events_for(Exl3TextContext& context) {
        auto& events = events_[&context];
        if (!events.ready) {
            check(cudaEventCreateWithFlags(&events.ready, cudaEventDisableTiming), "batched verify event");
            check(cudaEventCreateWithFlags(&events.done, cudaEventDisableTiming), "batched verify event");
        }
        return events;
    }

    const std::chrono::microseconds wait_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::map<const Exl3TextContext*, Events> events_;
    Pending* waiting_ = nullptr;
    int active_ = 0;
    std::uint64_t batched_ = 0, solo_ = 0, timeouts_ = 0, wait_us_ = 0, eager_ = 0;
};

}  // namespace ninfer::exl3
