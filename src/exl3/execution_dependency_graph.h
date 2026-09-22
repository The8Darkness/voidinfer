#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>

namespace ninfer::exl3 {

// Fixed request-local dependency description for the selected nondefault-
// stream DFlash slice. CUDA submission remains in the existing operators; this
// authority retains their owners and requires the final event which follows
// every command in the declared chain before any owner can be released.
class Exl3ExecutionDependencyGraph {
public:
    enum class Stage : std::uint8_t { draft,upload,attention,export_state,count };
    enum class Phase : std::uint8_t { idle,building,submitted,completed,cancelled,failed };
    struct Ticket {std::uint64_t acquisition=0,execution=0,generation=0;};
    struct Snapshot {
        Phase phase=Phase::idle;
        std::uint64_t generation=0;
        std::uintptr_t stream=0,event=0;
        std::uint8_t nodes=0,edges=0;
        int first_error=0;
        bool retains_owners=false;
    };

    Ticket begin(std::uint64_t acquisition,std::uint64_t execution,
        std::uintptr_t stream,std::shared_ptr<const void> stream_owner) {
        if(phase_!=Phase::idle || !acquisition || !execution || !stream ||
           !stream_owner || !stream_owner.use_count())
            throw std::invalid_argument(
                "execution dependency graph requires idle scoped nondefault stream");
        if(generation_==UINT64_MAX)
            throw std::overflow_error("execution dependency graph generation exhausted");
        acquisition_=acquisition;execution_=execution;stream_=stream;
        stream_owner_=std::move(stream_owner);event_=0;first_error_=0;
        for(auto& node:nodes_)node={};
        phase_=Phase::building;++generation_;
        return {acquisition,execution,generation_};
    }
    void add_node(Ticket ticket,Stage stage,std::shared_ptr<const void> owner) {
        require(ticket,Phase::building);
        const auto index=at(stage);auto& node=nodes_[index];
        if(node.owner || !owner || !owner.use_count())
            throw std::invalid_argument("execution dependency graph node owner");
        node.owner=std::move(owner);
    }
    void add_edge(Ticket ticket,Stage producer,Stage consumer) {
        require(ticket,Phase::building);
        const auto from=at(producer),to=at(consumer);
        if(from==to || !nodes_[from].owner || !nodes_[to].owner)
            throw std::invalid_argument("execution dependency graph edge endpoint");
        const auto bit=static_cast<std::uint8_t>(1u<<from);
        if(nodes_[to].dependencies&bit)
            throw std::invalid_argument("execution dependency graph duplicate edge");
        nodes_[to].dependencies|=bit;
        if(reaches(to,from)) {
            nodes_[to].dependencies&=static_cast<std::uint8_t>(~bit);
            throw std::invalid_argument("execution dependency graph cycle");
        }
    }
    void submit_selected_slice(Ticket ticket,std::uintptr_t event) {
        require(ticket,Phase::building);
        constexpr std::array<std::uint8_t,4> expected{
            0,1u<<0,1u<<1,1u<<2};
        for(std::size_t i=0;i<nodes_.size();++i)
            if(!nodes_[i].owner || nodes_[i].dependencies!=expected[i])
                throw std::logic_error("execution dependency graph missing selected edge");
        if(!event)throw std::invalid_argument(
            "execution dependency graph terminal event missing");
        event_=event;phase_=Phase::submitted;
    }
    bool complete(Ticket ticket,std::uintptr_t event,int error) noexcept {
        if(!current(ticket) || phase_!=Phase::submitted || !event || event!=event_)
            return false;
        if(error){first_error_=error;phase_=Phase::failed;return false;}
        phase_=Phase::completed;return true;
    }
    bool fail(Ticket ticket,int error) noexcept {
        if(!current(ticket) || phase_==Phase::idle || phase_==Phase::completed ||
           phase_==Phase::failed)return false;
        first_error_=error?error:-1;phase_=Phase::failed;return true;
    }
    bool cancel(Ticket ticket) noexcept {
        if(!current(ticket) || phase_!=Phase::building)return false;
        phase_=Phase::cancelled;return true;
    }
    bool retire(Ticket ticket) noexcept {
        if(!current(ticket) || phase_!=Phase::completed)return false;
        clear();return true;
    }
    // Caller has performed an enclosing stream/device drain. This is the only
    // failure cleanup path which may release a failed graph's retained owners.
    bool retire_after_stream_drain(Ticket ticket) noexcept {
        if(!current(ticket) || (phase_!=Phase::completed &&
           phase_!=Phase::cancelled && phase_!=Phase::failed))
            return false;
        clear();return true;
    }
    Snapshot snapshot() const noexcept {
        std::uint8_t nodes=0,edges=0;bool retains=bool(stream_owner_);
        for(const auto& node:nodes_)if(node.owner) {
            ++nodes;retains=true;
            auto mask=node.dependencies;
            while(mask){edges+=mask&1u;mask>>=1;}
        }
        return {phase_,generation_,stream_,event_,nodes,edges,first_error_,retains};
    }
private:
    struct Node {std::shared_ptr<const void> owner;std::uint8_t dependencies=0;};
    static constexpr std::size_t at(Stage stage) {
        const auto value=static_cast<std::size_t>(stage);
        if(value>=static_cast<std::size_t>(Stage::count))
            throw std::invalid_argument("execution dependency graph stage");
        return value;
    }
    bool current(Ticket ticket) const noexcept {
        return ticket.acquisition==acquisition_ && ticket.execution==execution_ &&
            ticket.generation==generation_ && ticket.generation;
    }
    void require(Ticket ticket,Phase phase) const {
        if(!current(ticket) || phase_!=phase)
            throw std::logic_error("execution dependency graph ticket/phase");
    }
    void clear() noexcept {
        for(auto& node:nodes_)node={};stream_owner_.reset();
        acquisition_=execution_=stream_=event_=0;first_error_=0;
        phase_=Phase::idle;
    }
    bool reaches(std::size_t from,std::size_t target) const noexcept {
        std::uint8_t seen=0,pending=static_cast<std::uint8_t>(1u<<from);
        while(pending) {
            unsigned node=0;while(!(pending&(1u<<node)))++node;
            pending&=static_cast<std::uint8_t>(~(1u<<node));
            if(node==target)return true;
            if(seen&(1u<<node))continue;seen|=static_cast<std::uint8_t>(1u<<node);
            for(std::size_t consumer=0;consumer<nodes_.size();++consumer)
                if(nodes_[consumer].dependencies&(1u<<node))
                    pending|=static_cast<std::uint8_t>(1u<<consumer);
        }
        return false;
    }
    std::array<Node,4> nodes_{};
    std::shared_ptr<const void> stream_owner_;
    Phase phase_=Phase::idle;
    std::uint64_t acquisition_=0,execution_=0,generation_=0;
    std::uintptr_t stream_=0,event_=0;
    int first_error_=0;
};

} // namespace ninfer::exl3
