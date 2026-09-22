#pragma once
#include "exl3/attention_stage_binding.h"
#include "exl3/registered_kv_upload.h"
#include "exl3/attention_history_coverage.h"

namespace ninfer::exl3 {
// Host-orchestrated asynchronous producer with explicit readiness/final-use
// edges. Waiting is deliberately separate from submission so current-bank
// compute can be queued while this producer is in flight.
class Exl3AttentionStageCommand {
    std::shared_ptr<Exl3AttentionStage> stages_;
    Exl3AttentionStageBinding binding_;
    Exl3KVUploadProvider provider_;
    unsigned failure_for_test_=0;
    Exl3AttentionStageCommand(std::shared_ptr<Exl3AttentionStage> stages,
        Exl3AttentionStageBinding binding,Exl3KVUploadProvider provider,unsigned failure_for_test=0)
        :stages_(std::move(stages)),binding_(binding),provider_(provider),failure_for_test_(failure_for_test){}
public:
    Exl3AttentionStageCommand(const Exl3AttentionStageCommand&)=delete;
    Exl3AttentionStageCommand& operator=(const Exl3AttentionStageCommand&)=delete;
    Exl3AttentionStageCommand(Exl3AttentionStageCommand&&)=default;
    // A witnessed producer with no consumer can retire on scope exit. Pending
    // producers and submitted consumers still require their actual event edge;
    // cancel preserves their retained owners and failed-transfer quarantine.
    ~Exl3AttentionStageCommand() {cancel();}
    using History=std::vector<std::shared_ptr<const Exl3ExactKVPage>>;
    // Admission is serialized by the context, like submit_history. Prove both
    // reserved slots and both represented planes before the first enqueue.
    // Transfer/provider failures after admission retain the existing quarantine.
    template<class HistoryType>
    static void require_history_pair(const std::shared_ptr<Exl3AttentionStage>& stages,
        const std::shared_ptr<HistoryType>& history,int bank,int position,
        const std::shared_ptr<Exl3AttentionStageStorage>& storage,std::uint64_t acquisition,
        std::uint64_t execution,
        const std::optional<std::array<Exl3KVRegistrationCache::ExternalRead,2>>& registration_reads={}) {
        if(!stages || !stages.use_count() || !history || !history.use_count() ||
            !storage || !storage.use_count() || position<1 || position>storage->capacity())
            throw std::invalid_argument("attention stage pair owner/capacity");
        exl3_require_attention_history(*history,position,bank);
        std::array<std::uintptr_t,4> events{};
        for(unsigned plane=0;plane<2;++plane) {
            if(registration_reads && !(*registration_reads)[plane].valid())
                throw std::invalid_argument("attention stage pair registration reader");
            events[plane*2]=reinterpret_cast<std::uintptr_t>(storage->producer_event(plane));
            events[plane*2+1]=reinterpret_cast<std::uintptr_t>(storage->consumer_event(plane));
            stages->require_producer(plane,acquisition,execution,events[plane*2],events[plane*2+1]);
        }
        for(unsigned i=0;i<events.size();++i)for(unsigned j=0;j<i;++j)
            if(events[i]==events[j])throw std::invalid_argument("attention stage pair event alias");
    }
    // History collection must be immutable and precredited by the caller. The
    // lease retains every page, including after command destruction on failure.
    template<class HistoryType>
    static Exl3AttentionStageCommand submit_history(std::shared_ptr<Exl3AttentionStage> stages,
        std::shared_ptr<HistoryType> history,int bank,Exl3ExactKVExtent::Plane plane,int position,
        const std::shared_ptr<Exl3AttentionStageStorage>& storage,std::uint64_t acquisition,
        std::uint64_t execution,cudaStream_t stream,Exl3KVUploadProvider provider={},unsigned failure_for_test=0,
        std::optional<Exl3KVRegistrationCache::ExternalRead> registration_read={},int first_row=0) {
        if(!stages || !stages.use_count() || !history || !history.use_count() ||
            !storage || !storage.use_count() || position<1 || position>storage->capacity() ||
            first_row<0 || first_row>position ||
            !provider.copy || !provider.record || !provider.wait || failure_for_test>5)
            throw std::invalid_argument("attention stage history owner/provider/capacity");
        exl3_require_attention_history(*history,position,bank);
        // The first complete page is retained only as the command's immutable
        // backing witness.  A suffix producer still addresses the reserved
        // stage from row zero, allowing a preceding device-prefix D2D copy and
        // the host suffix copies to share one producer event.
        auto first=Exl3ExactKVExtent::view(history->front(),bank,plane,position);
        auto binding=exl3_bind_attention_stage(*stages,std::move(first),storage,acquisition,execution,history,
            std::move(registration_read));
        auto failure=cudaErrorUnknown;
        try {
            for(const auto& page:*history) {
                const auto extent=Exl3ExactKVExtent::view(page,bank,plane,position,first_row);
                if(!extent.rows())continue;
                const auto error=failure_for_test==1?cudaErrorUnknown:provider.copy(binding.destination+extent.destination_offset(storage->capacity()),
                    extent.data(),extent.bytes(),cudaMemcpyHostToDevice,stream);
                if(error!=cudaSuccess) {failure=error;throw std::runtime_error("attention stage history copy failed");}
            }
            const auto error=failure_for_test==2?cudaErrorUnknown:provider.record(binding.producer,stream);
            if(error!=cudaSuccess) {failure=error;throw std::runtime_error("attention stage history record failed");}
        } catch(...) {
            stages->producer_finished(binding.ticket,reinterpret_cast<std::uintptr_t>(binding.producer),
                static_cast<int>(failure));
            throw;
        }
        return Exl3AttentionStageCommand(std::move(stages),binding,provider,failure_for_test);
    }
    static Exl3AttentionStageCommand submit(std::shared_ptr<Exl3AttentionStage> stages,
        Exl3ExactKVExtent extent,const std::shared_ptr<Exl3AttentionStageStorage>& storage,
        std::uint64_t acquisition,std::uint64_t execution,cudaStream_t stream,
        Exl3KVUploadProvider provider={}) {
        if(!stages || !stages.use_count() || !provider.copy || !provider.record || !provider.wait)
            throw std::invalid_argument("attention stage command owner/provider");
        const auto* source=extent.data();const auto bytes=extent.bytes();
        auto binding=exl3_bind_attention_stage(*stages,std::move(extent),storage,acquisition,execution);
        auto error=provider.copy(binding.destination,source,bytes,cudaMemcpyHostToDevice,stream);
        if(error==cudaSuccess)error=provider.record(binding.producer,stream);
        if(error!=cudaSuccess) {
            stages->producer_finished(binding.ticket,reinterpret_cast<std::uintptr_t>(binding.producer),static_cast<int>(error));
            throw std::runtime_error("attention stage producer submission failed");
        }
        return Exl3AttentionStageCommand(std::move(stages),binding,provider);
    }
    void await_producer() {
        const auto event=reinterpret_cast<std::uintptr_t>(binding_.producer);
        if(!stages_ || !stages_->producer_pending(binding_.ticket,event))
            throw std::logic_error("attention stage producer ticket no longer pending");
        const auto error=failure_for_test_==3?cudaErrorUnknown:provider_.wait(binding_.producer);
        if(!stages_->producer_finished(binding_.ticket,event,static_cast<int>(error)))
            throw std::runtime_error("attention stage producer completion failed");
    }
    std::uint16_t* consume(int bank,Exl3ExactKVExtent::Plane plane) {
        if(!stages_)throw std::logic_error("moved attention stage command");
        stages_->consume(binding_.ticket,bank,plane);return binding_.destination;
    }
    // The owning context serializes this schedule. Preflight the complete pair
    // before either slot transitions or the caller submits a consumer copy.
    static std::array<std::uint16_t*,2> consume_pair(Exl3AttentionStageCommand& key,
        Exl3AttentionStageCommand& value,int bank) {
        if(!key.stages_ || key.stages_!=value.stages_ || &key==&value)
            throw std::invalid_argument("attention stage consumer pair owner");
        key.stages_->require_consumer(key.binding_.ticket,bank,Exl3ExactKVExtent::Plane::key);
        value.stages_->require_consumer(value.binding_.ticket,bank,Exl3ExactKVExtent::Plane::value);
        return {key.consume(bank,Exl3ExactKVExtent::Plane::key),
            value.consume(bank,Exl3ExactKVExtent::Plane::value)};
    }
    void finish_consumer(cudaStream_t stream) {
        const auto event=reinterpret_cast<std::uintptr_t>(binding_.consumer);
        if(!stages_ || !stages_->consumer_pending(binding_.ticket,event))
            throw std::logic_error("attention stage consumer ticket no longer pending");
        auto error=failure_for_test_==4?cudaErrorUnknown:provider_.record(binding_.consumer,stream);
        if(error==cudaSuccess)error=failure_for_test_==5?cudaErrorUnknown:provider_.wait(binding_.consumer);
        if(!stages_->consumer_finished(binding_.ticket,event,static_cast<int>(error)))
            throw std::runtime_error("attention stage consumer completion failed");
    }
    bool cancel() noexcept {return stages_ && stages_->cancel(binding_.ticket);}
};
}
