#pragma once

#include "exl3/text_model.h"
#include "exl3/resource_inventory.h"
#include "exl3/bounded_shared_owner.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ninfer::exl3 {

// C2 transport-only rendezvous. Target reductions remain independent; this
// owner gathers two completed, explicitly sized decision ranges and performs
// one bounded D2H. No fake/padding row participates in the row map.
class Exl3EngineGreedyPacketBatch {
    struct Storage {
        Exl3GreedyRow* device=nullptr;
        Exl3GreedyRow* host=nullptr;
        cudaStream_t stream=nullptr;
        cudaEvent_t ready=nullptr;
        int first_error=0;
        Storage* next=nullptr;
    };
    inline static std::atomic<Storage*> quarantine_{nullptr};
    Storage* storage_=new Storage;
    struct Offer {
        Exl3TextContext& context;
        Exl3PendingGreedyPacket pending;
        Exl3GreedyBatchSource source;
        const std::atomic<bool>* cancelled;
        std::optional<Exl3GreedyPacket> result;
        std::exception_ptr failure;
        bool claimed=false,done=false;
        Offer(Exl3TextContext& c,Exl3PendingGreedyPacket&& p,
              Exl3GreedyBatchSource s,const std::atomic<bool>* x)
            :context(c),pending(std::move(p)),source(std::move(s)),cancelled(x){}
    };
    std::mutex mutex_;
    std::condition_variable changed_;
    Offer* waiting_=nullptr;
    bool busy_=false,failed_=false;
    std::exception_ptr first_failure_;
    std::chrono::microseconds timeout_{50};
    std::uint64_t batches_=0,batched_rows_=0,singles_=0,cancelled_=0,failures_=0;
    unsigned invalid_row_for_test_=0;
    std::function<void()> pair_observer_for_test_;

    static void check(cudaError_t error) {
        if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
    }
    static bool same_owner(const std::shared_ptr<const void>& a,
                           const std::shared_ptr<const void>& b) noexcept {
        return a && b && !a.owner_before(b) && !b.owner_before(a);
    }
    static bool compatible(const Offer& a,const Offer& b) noexcept {
        return a.source.owns_pending_execution() &&
            b.source.owns_pending_execution() &&
            &a.context!=&b.context && a.source.owner!=b.source.owner &&
            same_owner(a.source.model_identity,b.source.model_identity) &&
            a.source.rows>0 && b.source.rows>0 &&
            a.source.rows+b.source.rows<=16 &&
            a.source.device_row && b.source.device_row &&
            a.source.producer_ready && b.source.producer_ready;
    }
    static bool is_cancelled(const Offer& offer) noexcept {
        return offer.cancelled && offer.cancelled->load(std::memory_order_acquire);
    }
    static void cancelled(Offer& offer) noexcept {
        offer.failure=std::make_exception_ptr(
            std::runtime_error("batched greedy packet consumer cancelled before claim"));
    }
    void claim(Offer& offer) {
        *offer.source.consumer_stream=storage_->stream;
        bool unclaimed=false;
        if(!offer.source.consumer_claimed->compare_exchange_strong(
                unclaimed,true,std::memory_order_acq_rel))
            throw std::runtime_error("batched greedy packet source already claimed");
        offer.claimed=true;
    }
    void finish_one(Offer& offer,std::span<const Exl3GreedyRow> rows) noexcept {
        try {
            offer.result=offer.context.finish_batched_greedy_packet(
                std::move(offer.pending),rows,offer.source.acquisition,
                offer.source.execution);
        } catch(...) {offer.failure=std::current_exception();}
    }
    bool transport(Offer& first,Offer* second) noexcept {
        try {
            claim(first);if(second)claim(*second);
            check(cudaStreamWaitEvent(storage_->stream,first.source.producer_ready,0));
            check(cudaMemcpyAsync(storage_->device,first.source.device_row,
                first.source.rows*sizeof(Exl3GreedyRow),cudaMemcpyDeviceToDevice,storage_->stream));
            check(cudaEventRecord(first.source.consumer_done,storage_->stream));
            first.source.consumer_recorded->store(true,std::memory_order_release);
            if(second) {
                check(cudaStreamWaitEvent(storage_->stream,second->source.producer_ready,0));
                check(cudaMemcpyAsync(storage_->device+first.source.rows,second->source.device_row,
                    second->source.rows*sizeof(Exl3GreedyRow),cudaMemcpyDeviceToDevice,storage_->stream));
                check(cudaEventRecord(second->source.consumer_done,storage_->stream));
                second->source.consumer_recorded->store(true,std::memory_order_release);
            }
            const std::size_t rows=first.source.rows+(second?second->source.rows:0);
            check(cudaMemcpyAsync(storage_->host,storage_->device,
                rows*sizeof(Exl3GreedyRow),cudaMemcpyDeviceToHost,storage_->stream));
            check(cudaEventRecord(storage_->ready,storage_->stream));
            check(cudaEventSynchronize(storage_->ready));
            const auto invalid=std::exchange(invalid_row_for_test_,0u);
            if(invalid && invalid<=rows)storage_->host[invalid-1].nonfinite=1;
            finish_one(first,std::span<const Exl3GreedyRow>(storage_->host,first.source.rows));
            if(second)finish_one(*second,std::span<const Exl3GreedyRow>(
                storage_->host+first.source.rows,second->source.rows));
            return true;
        } catch(...) {
            auto failure=std::current_exception();
            if(!first.failure)first.failure=failure;
            if(second && !second->failure)second->failure=failure;
            return false;
        }
    }
    bool single(Offer& offer) noexcept {
        if(is_cancelled(offer)){cancelled(offer);return true;}
        return transport(offer,nullptr);
    }
public:
    struct Stats {
        std::uint64_t batches=0,batched_rows=0,singles=0,cancelled=0,failures=0;
    };
    static constexpr std::size_t device_bytes() noexcept {
        return 16*sizeof(Exl3GreedyRow);
    }
    static constexpr std::size_t registered_host_bytes() noexcept {
        return 16*sizeof(Exl3GreedyRow);
    }
    static constexpr std::size_t host_metadata_bytes() noexcept {
        return bounded_shared_allocation_bytes<Exl3EngineGreedyPacketBatch>()+
            sizeof(Storage);
    }
    static Exl3ResourceInventory::Requirement requirement() {
        Exl3ResourceInventory::Requirement result;result.configuration=0x47504232;
        result.add(Exl3ResourceInventory::Domain::device,1,device_bytes());
        result.add(Exl3ResourceInventory::Domain::cuda_registered_host,1,registered_host_bytes());
        result.add(Exl3ResourceInventory::Domain::host_metadata,1,host_metadata_bytes());
        return result;
    }
    Exl3EngineGreedyPacketBatch() {
        try {
            check(cudaMalloc(reinterpret_cast<void**>(&storage_->device),device_bytes()));
            check(cudaHostAlloc(reinterpret_cast<void**>(&storage_->host),registered_host_bytes(),
                cudaHostAllocPortable));
            check(cudaStreamCreateWithFlags(&storage_->stream,cudaStreamNonBlocking));
            check(cudaEventCreateWithFlags(&storage_->ready,cudaEventDisableTiming));
        } catch(...) {retire();throw;}
    }
    Exl3EngineGreedyPacketBatch(const Exl3EngineGreedyPacketBatch&)=delete;
    Exl3EngineGreedyPacketBatch& operator=(const Exl3EngineGreedyPacketBatch&)=delete;
    ~Exl3EngineGreedyPacketBatch(){retire();}
    // Owning Engine has joined both producers and completed its device drain.
    void retire_after_device_drain() noexcept {retire();}
    void retire() noexcept {
        if(!storage_)return;
        int error=0;
        if(storage_->stream)error=static_cast<int>(cudaStreamSynchronize(storage_->stream));
        if(!error && storage_->ready)error=static_cast<int>(cudaEventDestroy(storage_->ready));
        if(!error){storage_->ready=nullptr;if(storage_->stream)error=static_cast<int>(cudaStreamDestroy(storage_->stream));}
        if(!error){storage_->stream=nullptr;if(storage_->host)error=static_cast<int>(cudaFreeHost(storage_->host));}
        if(!error){storage_->host=nullptr;if(storage_->device)error=static_cast<int>(cudaFree(storage_->device));}
        if(!error){delete std::exchange(storage_,nullptr);return;}
        storage_->first_error=error;auto* retained=std::exchange(storage_,nullptr);
        auto* head=quarantine_.load(std::memory_order_relaxed);
        do{retained->next=head;}while(!quarantine_.compare_exchange_weak(
            head,retained,std::memory_order_release,std::memory_order_relaxed));
    }
    static bool retirement_unresolved() noexcept {
        return quarantine_.load(std::memory_order_acquire)!=nullptr;
    }
    Exl3GreedyPacket finish(Exl3TextContext& context,
        Exl3PendingGreedyPacket&& pending,std::uint64_t acquisition,
        std::uint64_t execution,const std::atomic<bool>* cancelled_flag=nullptr) {
        auto source=context.greedy_batch_source(pending,acquisition,execution);
        if(!source.owns_pending_execution())
            throw std::runtime_error("batched greedy packet missing final-use owners");
        Offer offer(context,std::move(pending),std::move(source),cancelled_flag);
        std::unique_lock<std::mutex> lock(mutex_);
        if(failed_)std::rethrow_exception(first_failure_);
        if(!busy_ && waiting_ && compatible(*waiting_,offer) &&
           !is_cancelled(*waiting_) && !is_cancelled(offer)) {
            auto* peer=waiting_;waiting_=nullptr;busy_=true;
            peer->claimed=true;offer.claimed=true;
            auto pair_observer=std::move(pair_observer_for_test_);lock.unlock();
            if(pair_observer)pair_observer();
            const bool completed=transport(*peer,&offer);lock.lock();
            ++batches_;batched_rows_+=peer->source.rows+offer.source.rows;
            peer->done=true;offer.done=true;
            if(!completed){failed_=true;first_failure_=offer.failure;++failures_;}
            busy_=false;changed_.notify_all();
        } else if(!busy_ && !waiting_ && !is_cancelled(offer)) {
            waiting_=&offer;
            changed_.wait_for(lock,timeout_,[&]{return offer.done || offer.claimed ||
                is_cancelled(offer) || failed_;});
            if(!offer.done && !offer.claimed) {
                if(waiting_==&offer)waiting_=nullptr;
                const bool was_cancelled=is_cancelled(offer);
                busy_=true;
                lock.unlock();const bool completed=single(offer);lock.lock();offer.done=true;
                if(was_cancelled)++cancelled_;else ++singles_;
                if(!completed){failed_=true;first_failure_=offer.failure;++failures_;}
                busy_=false;changed_.notify_all();
            } else while(!offer.done)changed_.wait(lock);
        } else {
            while(busy_ && !failed_)changed_.wait(lock);
            if(failed_)std::rethrow_exception(first_failure_);
            const bool was_cancelled=is_cancelled(offer);
            busy_=true;
            lock.unlock();const bool completed=single(offer);lock.lock();offer.done=true;
            if(was_cancelled)++cancelled_;else ++singles_;
            if(!completed){failed_=true;first_failure_=offer.failure;++failures_;}
            busy_=false;changed_.notify_all();
        }
        if(offer.failure)std::rethrow_exception(offer.failure);
        if(!offer.result)throw std::runtime_error("batched greedy packet missing result");
        return std::move(*offer.result);
    }
    Stats stats() {
        std::lock_guard<std::mutex> lock(mutex_);return {batches_,batched_rows_,singles_,cancelled_,failures_};
    }
    void notify_cancellation() {
        std::lock_guard<std::mutex> lock(mutex_);
        if(waiting_ && is_cancelled(*waiting_))changed_.notify_all();
    }
    void set_timeout_for_test(std::chrono::microseconds timeout) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(waiting_ || busy_ || timeout.count()<1 || timeout>std::chrono::milliseconds(5))
            throw std::invalid_argument("greedy packet batch timeout requires idle 1us..5ms");
        timeout_=timeout;
    }
    void invalidate_next_row_for_test(unsigned one_based_row) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(waiting_ || busy_ || invalid_row_for_test_ || one_based_row<1 || one_based_row>16)
            throw std::invalid_argument("greedy packet invalid-row seam requires idle row1..16");
        invalid_row_for_test_=one_based_row;
    }
    void observe_next_pair_for_test(std::function<void()> observer) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!observer || waiting_ || busy_ || pair_observer_for_test_)
            throw std::invalid_argument("greedy packet pair observer requires idle owner");
        pair_observer_for_test_=std::move(observer);
    }
};

} // namespace ninfer::exl3
