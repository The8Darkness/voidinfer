#pragma once
#include "exl3/prefill_chunk_contract.h"
#include "exl3/exact_outer_reference.h"
#include "exl3/dflash2_draft.h"
#include "exl3/token_history.h"
#include "exl3/prepared_identity.h"
#include "exl3/resource_inventory.h"
#include "exl3/bounded_shared_owner.h"
#include "exl3/reserved_host_payload_union.h"
#include "exl3/pending_payload_extent.h"
#include "exl3/encoded_media_cache.h"
#include <ninfer/targets/qwen3_6/vision_control.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <unordered_set>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <atomic>
#include <condition_variable>
#include <thread>

namespace ninfer::exl3 {
class Exl3VeriCacheReplay;
class Exl3PreparedMediaReplay;
class Exl3VeriCachePrefixIndex;
class Exl3VisionContext;

struct Exl3MediaTapOrigin {
    std::shared_ptr<const Exl3PreparedIdentity> prepared_identity;
    std::shared_ptr<const void> request_revision;
    std::shared_ptr<const Exl3ExactHostState> target_state;
    std::vector<std::shared_ptr<const void>> tap_blocks;
    int position=0;
};

// Immutable authoritative request checkpoint. The bounded draft conditioning
// window and L2 state are published together; no public constructor can pair a
// different request's taps with this state. Callers serialize the physical lanes.
class Exl3VeriCacheRequest {
    struct ConstructionKey {};
    struct TapBlock {
        std::optional<RetainedHostAllocationLedger::Ticket> payload_credit;
        int first=0, rows=0;
        std::array<std::vector<std::uint16_t>,5> taps;
        TapBlock()=default;
        TapBlock(int start,int count,std::array<std::vector<std::uint16_t>,5> values,
            std::optional<RetainedHostAllocationLedger::Ticket> credit={})
            :payload_credit(std::move(credit)),first(start),rows(count),taps(std::move(values)) {}
    };
    std::shared_ptr<const Exl3ExactHostState> state_;
    static std::shared_ptr<const int> create_revision(
        const Exl3TextContext::SnapshotMetadataReservation& reserve={}) {
        const auto bytes=bounded_shared_allocation_bytes<int>();
        std::optional<RetainedDescriptorLedger::Ticket> credit;
        if(reserve) {
            credit.emplace(reserve(bytes));
            if(credit->bytes()!=bytes)throw std::invalid_argument("request revision metadata extent");
        }
        auto result=make_bounded_shared<int>(0);
        if(credit && !attach_bounded_split_retirement_credit<int>(result,std::move(*credit),0))
            throw std::logic_error("request revision metadata attachment");
        return result;
    }
    std::shared_ptr<const int> revision_=create_revision(), parent_revision_;
    std::shared_ptr<const int> replay_anchor_revision_;
    Exl3TokenHistory history_;
    std::shared_ptr<const Exl3PreparedIdentity> prepared_identity_;
    std::shared_ptr<const Exl3MediaTapOrigin> media_tap_origin_;
    std::vector<std::shared_ptr<const TapBlock>> blocks_;
    std::vector<std::pair<int,int>> partitions_;
    bool descriptor_capacity_locked_=false;
    std::shared_ptr<const Exl3DraftHostRing> projected_;
    Exl3VeriCacheRequest() = default;
    Exl3VeriCacheRequest(const Exl3VeriCacheRequest&)=default;
    Exl3VeriCacheRequest& operator=(const Exl3VeriCacheRequest&)=delete;
    using MetadataReservation=std::function<RetainedDescriptorLedger::Ticket(std::uint64_t)>;
    static std::shared_ptr<Exl3VeriCacheRequest> create(const MetadataReservation& reserve={}) {
        std::optional<RetainedDescriptorLedger::Ticket> credit;
        const auto bytes=bounded_shared_allocation_bytes<Exl3VeriCacheRequest>();
        if(reserve) {
            credit.emplace(reserve(bytes));
            if(credit->bytes()!=bytes)throw std::invalid_argument("request construction metadata extent");
        }
        auto result=make_bounded_shared<Exl3VeriCacheRequest>(ConstructionKey{});
        if(credit && credit->bytes()!=result->request_metadata_bytes())
            throw std::logic_error("request construction allocation extent");
        if(credit && !attach_request_metadata_credit(result,std::move(*credit)))
            throw std::logic_error("request construction metadata attachment");
        result->descriptor_capacity_locked_=static_cast<bool>(reserve);
        return result;
    }
    static std::shared_ptr<Exl3VeriCacheRequest> clone(const Exl3VeriCacheRequest& source,
        const MetadataReservation& reserve={}) {
        // Copy construction owns new vector storage, not the source's spare
        // capacity. Refuse a differing allocator extent before publication.
        std::optional<RetainedDescriptorLedger::Ticket> credit;
        if(reserve) {
            Exl3ResourceInventory::Requirement required;using Domain=Exl3ResourceInventory::Domain;
            required.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<Exl3VeriCacheRequest>());
            if(source.blocks_.size())required.add(Domain::host_metadata,source.blocks_.size(),sizeof(source.blocks_[0]));
            if(source.partitions_.size())required.add(Domain::host_metadata,source.partitions_.size(),sizeof(source.partitions_[0]));
            const auto bytes=required.units[static_cast<unsigned>(Domain::host_metadata)];
            credit.emplace(reserve(bytes));
            if(credit->bytes()!=bytes)throw std::invalid_argument("request clone metadata extent");
        }
        auto result=make_bounded_shared<Exl3VeriCacheRequest>(ConstructionKey{},source);
        if(credit && credit->bytes()!=result->request_metadata_bytes())
            throw std::logic_error("request clone allocation extent");
        if(credit && !attach_request_metadata_credit(result,std::move(*credit)))
            throw std::logic_error("request clone metadata attachment");
        result->descriptor_capacity_locked_=static_cast<bool>(reserve);
        return result;
    }
    friend class Exl3VeriCacheReplay;
    void bind_media_tap_origin(const MetadataReservation& reserve={}) {
        if(!prepared_identity_) {media_tap_origin_.reset();return;}
        if(!state_ || !revision_)throw std::logic_error("media tap origin authority missing");
        std::vector<std::shared_ptr<const void>> taps;taps.reserve(blocks_.size());
        for(const auto& block:blocks_)taps.emplace_back(block);
        const auto dynamic=taps.capacity()*sizeof(taps[0]);
        const auto bytes=bounded_shared_allocation_bytes<Exl3MediaTapOrigin>()+dynamic;
        std::optional<RetainedDescriptorLedger::Ticket> credit;
        if(reserve) {credit.emplace(reserve(bytes));if(credit->bytes()!=bytes)
            throw std::invalid_argument("media tap origin metadata extent");}
        auto origin=make_bounded_shared<Exl3MediaTapOrigin>(Exl3MediaTapOrigin{
            prepared_identity_,revision_,state_,std::move(taps),state_->position()});
        if(credit && !attach_bounded_split_retirement_credit<Exl3MediaTapOrigin>(origin,
            std::move(*credit),dynamic))throw std::logic_error("media tap origin metadata attachment");
        media_tap_origin_=std::move(origin);
    }
    static std::shared_ptr<const Exl3VeriCacheRequest> initialize_prepared_numeric_data(
        Exl3TextContext&,Exl3VisionContext&,
        const targets::qwen3_6::PreparedPromptData&,
        const targets::qwen3_6::VisionControl*,
        std::shared_ptr<const Exl3VeriCacheRequest>,const std::function<bool()>&,
        const Exl3EncodedMediaRetentionReserve&);

    static std::size_t planned_descriptor_count(std::size_t existing,std::size_t incoming_rows) {
        const auto limit=static_cast<std::size_t>(Exl3Dflash2DraftModel::ring_capacity())+1;
        if(existing>limit)throw std::logic_error("request descriptor ring extent");
        return existing+std::min(incoming_rows,limit-existing);
    }
    static std::shared_ptr<Exl3VeriCacheRequest> create_planned(
        const Exl3VeriCacheRequest* source,std::size_t incoming_rows,bool retain_blocks,
        const MetadataReservation& reserve={},const MetadataReservation& revision_reserve={}) {
        const auto blocks=planned_descriptor_count(source && retain_blocks?source->blocks_.size():0,incoming_rows);
        const auto partitions=planned_descriptor_count(source?source->partitions_.size():0,incoming_rows);
        Exl3ResourceInventory::Requirement required;using Domain=Exl3ResourceInventory::Domain;
        required.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<Exl3VeriCacheRequest>());
        if(blocks)required.add(Domain::host_metadata,blocks,sizeof(std::shared_ptr<const TapBlock>));
        if(partitions)required.add(Domain::host_metadata,partitions,sizeof(std::pair<int,int>));
        const auto bytes=required.units[static_cast<unsigned>(Domain::host_metadata)];
        // Declared first so failed construction releases storage before its promise.
        std::optional<RetainedDescriptorLedger::Ticket> credit;
        if(reserve) {
            credit.emplace(reserve(bytes));
            if(credit->bytes()!=bytes)throw std::invalid_argument("planned request metadata extent");
        }
        auto revision=create_revision(revision_reserve);
        auto result=make_bounded_shared<Exl3VeriCacheRequest>(ConstructionKey{},std::move(revision));
        result->blocks_.reserve(blocks);result->partitions_.reserve(partitions);
        if(source) {
            if(retain_blocks)result->blocks_.insert(result->blocks_.end(),source->blocks_.begin(),source->blocks_.end());
            result->partitions_.insert(result->partitions_.end(),source->partitions_.begin(),source->partitions_.end());
            result->media_tap_origin_=source->media_tap_origin_;
        }
        if(credit && credit->bytes()!=result->request_metadata_bytes())
            throw std::logic_error("planned request allocation extent");
        if(credit && !attach_request_metadata_credit(result,std::move(*credit)))
            throw std::logic_error("planned request metadata attachment");
        result->descriptor_capacity_locked_=static_cast<bool>(reserve);
        return result;
    }

    // Each retained partition contains at least one row. Include the temporary
    // new entry inserted before eviction of the partition crossing the ring.
    void reserve_prompt_descriptors(std::size_t incoming_rows) {
        if(descriptor_capacity_locked_)throw std::logic_error("credited request descriptor replanning");
        blocks_.reserve(planned_descriptor_count(blocks_.size(),incoming_rows));
        partitions_.reserve(planned_descriptor_count(partitions_.size(),incoming_rows));
    }

    void append_partition(int first,int rows) {
        if(descriptor_capacity_locked_ && partitions_.size()==partitions_.capacity())
            throw std::logic_error("credited request partition capacity exhausted");
        if(rows<1 || rows>16 || first<0 || first>std::numeric_limits<int>::max()-rows || (!partitions_.empty() &&
            partitions_.back().first+partitions_.back().second!=first))
            throw std::invalid_argument("authoritative draft block continuity");
        partitions_.emplace_back(first,rows);
        const int oldest=first+rows-Exl3Dflash2DraftModel::ring_capacity();
        while(partitions_.size()>1 && partitions_.front().first+partitions_.front().second<=oldest)
            partitions_.erase(partitions_.begin());
    }
    void append(int first,int rows,std::array<std::vector<std::uint16_t>,5> incoming_taps,
        const MetadataReservation& reserve={},std::optional<RetainedHostAllocationLedger::Ticket> incoming_credit={},
        const Exl3TextContext::RequestHostPayloadObserver& observe={}) {
        // On every failure, release moved payload before its charge. Do not
        // depend on the unspecified destruction order of function arguments.
        auto payload_credit=std::move(incoming_credit);
        auto taps=std::move(incoming_taps);
        if(descriptor_capacity_locked_ && (blocks_.size()==blocks_.capacity() || partitions_.size()==partitions_.capacity()))
            throw std::logic_error("credited request tap descriptor capacity exhausted");
        if(rows<1 || rows>16 || first<0 || first>std::numeric_limits<int>::max()-rows || (!partitions_.empty() &&
            partitions_.back().first+partitions_.back().second!=first))
            throw std::invalid_argument("authoritative draft block continuity");
        for(const auto& layer:taps) if(layer.size()!=static_cast<std::size_t>(rows)*5120)
            throw std::invalid_argument("authoritative draft tap extent");
        if(payload_credit) {
            std::array<std::size_t,5> capacities{};
            for(std::size_t plane=0;plane<capacities.size();++plane)capacities[plane]=taps[plane].capacity();
            const auto actual=pending_payload_extent(0,capacities);
            if(!actual || payload_credit->bytes()!=*actual)
                throw std::invalid_argument("authoritative tap payload credit extent");
        }
        std::optional<RetainedDescriptorLedger::Ticket> credit;
        if(reserve) {
            credit.emplace(reserve(tap_block_metadata_bytes()));
            if(credit->bytes()!=tap_block_metadata_bytes())throw std::invalid_argument("tap block metadata extent");
        }
        auto block=make_bounded_shared<TapBlock>(first,rows,std::move(taps),std::move(payload_credit));
        if(credit && !attach_tap_block_metadata_credit(block,std::move(*credit)))
            throw std::logic_error("tap block metadata attachment");
        if(block->payload_credit && observe) {
            const std::shared_ptr<const void> owner=block;
            for(std::size_t plane=0;plane<5;++plane)
                observe(owner,plane,block->taps[plane].data(),block->taps[plane].capacity()*sizeof(std::uint16_t));
        }
        blocks_.push_back(std::move(block));
        // Keep the complete call crossing the ring boundary. Repartitioning a
        // retained call could change projection dispatch and rounding.
        const int oldest=first+rows-Exl3Dflash2DraftModel::ring_capacity();
        while(blocks_.size()>1 && blocks_.front()->first+blocks_.front()->rows<=oldest)
            blocks_.erase(blocks_.begin());
        append_partition(first,rows);
    }
    static void validate_prefill_width(int width) {
        exl3_require_prefill_width(width);
    }
    void ingest_prompt_suffix(Exl3TextContext& exact,std::span<const std::int64_t> suffix,
        int width,const std::function<void(int)>& progress,
        const std::function<void(int,int,const std::array<std::vector<std::uint16_t>,5>&)>& fresh={},
        bool actual_tail=false) {
        const auto retain=[&](int first,int rows,std::array<std::vector<std::uint16_t>,5> taps) {
            if(fresh)fresh(first,rows,taps);
            append(first,rows,std::move(taps),exact.request_metadata_reservation());
        };
        for(std::size_t offset=0;offset<suffix.size();) {
            const auto step=exl3_prefill_chunk_step(width,suffix.size()-offset,exact.position(),exact.max_context(),actual_tail);
            const int rows=step.rows;
            const int first=step.first;
            if(actual_tail && rows>=17 && rows<=31) exact.append_exact_prefill_tail(suffix.subspan(offset,rows));
            else if(rows>8) exact.append_exact_prefill_wide(suffix.subspan(offset,rows));
            else if(rows>1) exact.continue_rows(suffix.subspan(offset,rows));
            else exact.decode(suffix[offset]);
            auto taps=exact.exact_tap_rows_host();
            // Target width never changes the original eight-row draft commit
            // partitions. Initial16 remains its separate original call.
            if(rows<=8) retain(first,rows,std::move(taps));
            else for(int start=0;start<rows;start+=8) {
                const int count=std::min(8,rows-start);
                std::array<std::vector<std::uint16_t>,5> block;
                for(int tap=0;tap<5;++tap)
                    block[tap].assign(taps[tap].begin()+start*5120,taps[tap].begin()+(start+count)*5120);
                retain(first+start,count,std::move(block));
            }
            if(rows>8) exact.finish_exact_prefill();
            else if(rows>1) exact.finish_exact_continuation();
            if(exact.position()!=step.next_position)
                throw std::logic_error("prefill caller changed planned absolute row sequence");
            offset+=rows;
            if(progress) progress(exact.position());
        }
    }

public:
    explicit Exl3VeriCacheRequest(ConstructionKey):Exl3VeriCacheRequest() {}
    Exl3VeriCacheRequest(ConstructionKey,std::shared_ptr<const int> revision):revision_(std::move(revision)) {}
    Exl3VeriCacheRequest(ConstructionKey,const Exl3VeriCacheRequest& source):Exl3VeriCacheRequest(source) {
        descriptor_capacity_locked_=false;
    }
    std::uint64_t request_metadata_bytes() const {
        Exl3ResourceInventory::Requirement required;using Domain=Exl3ResourceInventory::Domain;
        required.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<Exl3VeriCacheRequest>());
        if(blocks_.capacity())required.add(Domain::host_metadata,blocks_.capacity(),sizeof(blocks_[0]));
        if(partitions_.capacity())required.add(Domain::host_metadata,partitions_.capacity(),sizeof(partitions_[0]));
        return required.units[static_cast<unsigned>(Domain::host_metadata)];
    }
    enum class MetadataOwnerKind : std::uint8_t {
        request,revision,prepared_identity,media_tap_origin,draft_ring,draft_page,exact_page,
        tap_block,token_node,exact_state,count
    };
    using MetadataMatches=bool(*)(const std::shared_ptr<const void>&,
        const RetainedDescriptorLedger&) noexcept;
    using MetadataAttach=bool(*)(const std::shared_ptr<const void>&,
        RetainedDescriptorLedger::Ticket) noexcept;
    struct MetadataAllocation {
        std::shared_ptr<const void> owner;
        std::uint64_t bytes=0;
        MetadataOwnerKind kind=MetadataOwnerKind::request;
        MetadataMatches matches=nullptr;
        MetadataAttach attach=nullptr;
    };
    struct MetadataAllocationStats {
        std::uint64_t logical_bytes=0,unique_owners=0;
        std::array<std::uint64_t,static_cast<std::size_t>(MetadataOwnerKind::count)> bytes_by_kind{};
    };
    static std::uint64_t checked_metadata_sum_for_test(std::uint64_t current,
        std::uint64_t bytes) {
        if(bytes>UINT64_MAX-current)
            throw std::overflow_error("request metadata allocation accounting overflow");
        return current+bytes;
    }
    static bool request_metadata_credit_belongs_to(const std::shared_ptr<const void>& owner,
        const RetainedDescriptorLedger& ledger) noexcept {
        return bounded_split_credit_belongs_to<Exl3VeriCacheRequest>(owner,ledger);
    }
    static bool attach_request_metadata_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        if(!owner || !owner.use_count())return false;
        const auto* request=static_cast<const Exl3VeriCacheRequest*>(owner.get());
        const auto blocks=request->blocks_.capacity()*sizeof(request->blocks_[0]);
        const auto partitions=request->partitions_.capacity()*sizeof(request->partitions_[0]);
        if(partitions>std::numeric_limits<std::size_t>::max()-blocks)return false;
        return attach_bounded_split_retirement_credit<Exl3VeriCacheRequest>(owner,std::move(credit),blocks+partitions);
    }
    const Exl3PreparedIdentity* prepared_identity() const noexcept {return prepared_identity_.get();}
    const std::shared_ptr<const Exl3PreparedIdentity>& prepared_metadata_owner() const noexcept {return prepared_identity_;}
    std::shared_ptr<const void> revision_owner() const noexcept{return revision_;}
    const std::shared_ptr<const Exl3MediaTapOrigin>& media_tap_origin() const noexcept{return media_tap_origin_;}
    bool media_tap_origin_current() const noexcept {
        if(!prepared_identity_)return !media_tap_origin_;
        if(!media_tap_origin_ || !media_tap_origin_->prepared_identity ||
           !media_tap_origin_->request_revision || !media_tap_origin_->target_state ||
           !state_ || media_tap_origin_->position!=media_tap_origin_->target_state->position() ||
           media_tap_origin_->position>state_->position() || media_tap_origin_->tap_blocks.empty() ||
           !media_tap_origin_->prepared_identity->prefix_positions_equal(
               *prepared_identity_,static_cast<std::size_t>(media_tap_origin_->position)))return false;
        return std::all_of(media_tap_origin_->tap_blocks.begin(),media_tap_origin_->tap_blocks.end(),
            [](const auto& owner){return owner && owner.use_count();});
    }
    bool same_media_tap_origin(const Exl3VeriCacheRequest& other) const noexcept {
        return media_tap_origin_ && media_tap_origin_.get()==other.media_tap_origin_.get();
    }
    bool owns_media_tap_origin(const std::shared_ptr<const Exl3MediaTapOrigin>& origin) const noexcept {
        return media_tap_origin_current() && origin && origin.get()==media_tap_origin_.get();
    }
    std::uint64_t media_tap_origin_metadata_bytes() const noexcept {
        return media_tap_origin_?bounded_shared_allocation_bytes<Exl3MediaTapOrigin>()+
            media_tap_origin_->tap_blocks.capacity()*sizeof(media_tap_origin_->tap_blocks[0]):0;
    }
    static bool media_tap_origin_credit_belongs_to(const std::shared_ptr<const void>& owner,
        const RetainedDescriptorLedger& ledger) noexcept {
        return bounded_split_credit_belongs_to<Exl3MediaTapOrigin>(owner,ledger);
    }
    static bool attach_media_tap_origin_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        if(!owner || !owner.use_count())return false;
        const auto* origin=static_cast<const Exl3MediaTapOrigin*>(owner.get());
        return attach_bounded_split_retirement_credit<Exl3MediaTapOrigin>(owner,std::move(credit),
            origin->tap_blocks.capacity()*sizeof(origin->tap_blocks[0]));
    }
    template<class Visitor> void visit_token_control_owners(Visitor&& visitor) const {
        history_.visit_control_owners(std::forward<Visitor>(visitor));
    }
    static constexpr std::size_t tap_block_metadata_bytes() noexcept {
        // Vector objects are inside TapBlock; their FP16 allocations are payload.
        return bounded_shared_allocation_bytes<TapBlock>();
    }
    template<class Visitor> void visit_tap_block_owners(Visitor&& visitor) const {
        for(const auto& block:blocks_)visitor(std::shared_ptr<const void>(block));
    }
    template<class Visitor> void visit_reserved_tap_payloads(const RetainedHostAllocationLedger& ledger,
        Visitor&& visitor) const {
        // Report each credited allocation with its immutable shared owner.
        // Cross-root callers deduplicate owner + plane before page rounding.
        for(const auto& block:blocks_) {
            if(!block->payload_credit || !ledger.owns(*block->payload_credit))continue;
            for(std::size_t plane=0;plane<block->taps.size();++plane) {
                const auto& values=block->taps[plane];
                if(values.capacity())visitor(std::shared_ptr<const void>(block),plane,
                    values.data(),values.capacity()*sizeof(std::uint16_t));
            }
        }
    }
    template<std::size_t Capacity> static ReservedHostPayloadUnion<Capacity> reserved_tap_payload_union(
        std::span<const std::shared_ptr<const Exl3VeriCacheRequest>> roots,
        const RetainedHostAllocationLedger& ledger) {
        ReservedHostPayloadUnion<Capacity> result;
        for(const auto& root:roots) {
            if(!root || !root.use_count())throw std::invalid_argument("reserved tap union requires owned roots");
            root->visit_reserved_tap_payloads(ledger,[&](auto owner,std::size_t plane,const void* pointer,std::size_t bytes) {
                result.add(std::move(owner),plane,pointer,bytes);
            });
        }
        return result;
    }
    // Owner must come from tap-block construction/visitation, not an arbitrary
    // erased pointer. This checks the ledger and exact immutable plane extent.
    static bool reserved_tap_payload_matches(const std::shared_ptr<const void>& owner,
        const RetainedHostAllocationLedger& ledger,std::size_t plane,const void* data,std::size_t bytes) noexcept {
        if(!owner || !owner.use_count() || plane>=5)return false;
        const auto* block=static_cast<const TapBlock*>(owner.get());
        return block->payload_credit && ledger.owns(*block->payload_credit) &&
            data==block->taps[plane].data() && bytes==block->taps[plane].capacity()*sizeof(std::uint16_t);
    }
    static bool tap_block_metadata_credit_belongs_to(const std::shared_ptr<const void>& owner,
        const RetainedDescriptorLedger& ledger) noexcept {
        return bounded_split_credit_belongs_to<TapBlock>(owner,ledger);
    }
    static bool attach_tap_block_metadata_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        return attach_bounded_split_retirement_credit<TapBlock>(owner,std::move(credit),0);
    }
    template<class Visitor> void visit_revision_owners(Visitor&& visitor) const {
        const std::array owners{revision_,parent_revision_,replay_anchor_revision_};
        for(std::size_t i=0;i<owners.size();++i) {
            if(!owners[i])continue;
            bool seen=false;
            for(std::size_t j=0;j<i;++j)seen|=owners[j]==owners[i];
            if(!seen)visitor(owners[i]);
        }
    }
    static void visit_metadata_allocations(
        std::span<const std::shared_ptr<const Exl3VeriCacheRequest>> roots,
        const std::function<void(const MetadataAllocation&)>& visitor) {
        if(!visitor)throw std::invalid_argument("request metadata allocation visitor missing");
        for(const auto& root:roots)
            if(!root || !root->state_)throw std::invalid_argument(
                "request metadata allocation root/state missing");
        struct Key {
            const void* address=nullptr;MetadataOwnerKind kind=MetadataOwnerKind::request;
            bool operator==(const Key&) const noexcept=default;
        };
        struct Hash {
            std::size_t operator()(const Key& key) const noexcept {
                return std::hash<const void*>{}(key.address)^(
                    static_cast<std::size_t>(key.kind)*0x9e3779b97f4a7c15ULL);
            }
        };
        std::unordered_set<Key,Hash> seen;
        std::vector<MetadataAllocation> allocations;
        const auto append=[&](std::shared_ptr<const void> owner,std::uint64_t bytes,
            MetadataOwnerKind kind,MetadataMatches matches,MetadataAttach attach) {
            if(!owner || !owner.use_count() || !bytes || !matches || !attach)
                throw std::logic_error("request metadata allocation identity/extent");
            if(seen.insert({owner.get(),kind}).second)
                allocations.push_back({std::move(owner),bytes,kind,matches,attach});
        };
        const auto revision_matches=+[](const std::shared_ptr<const void>& owner,
            const RetainedDescriptorLedger& ledger) noexcept {
            return bounded_split_credit_belongs_to<int>(owner,ledger);
        };
        const auto revision_attach=+[](const std::shared_ptr<const void>& owner,
            RetainedDescriptorLedger::Ticket credit) noexcept {
            return attach_bounded_split_retirement_credit<int>(owner,std::move(credit),0);
        };
        for(const auto& root:roots) {
            append(root,root->request_metadata_bytes(),MetadataOwnerKind::request,
                &request_metadata_credit_belongs_to,&attach_request_metadata_credit);
            root->visit_revision_owners([&](const auto& revision) {
                append(revision,bounded_shared_allocation_bytes<int>(),
                    MetadataOwnerKind::revision,revision_matches,revision_attach);
            });
            if(const auto& identity=root->prepared_identity_)
                append(identity,identity->metadata_bytes(),MetadataOwnerKind::prepared_identity,
                    &Exl3PreparedIdentity::metadata_credit_belongs_to,
                    &Exl3PreparedIdentity::attach_metadata_credit);
            if(const auto& origin=root->media_tap_origin_)
                append(origin,root->media_tap_origin_metadata_bytes(),MetadataOwnerKind::media_tap_origin,
                    &media_tap_origin_credit_belongs_to,&attach_media_tap_origin_credit);
            if(const auto& ring=root->projected_) {
                append(ring,ring->metadata_bytes(),MetadataOwnerKind::draft_ring,
                    &Exl3DraftHostRing::metadata_credit_belongs_to,
                    &Exl3DraftHostRing::attach_metadata_credit);
                ring->visit_page_metadata_owners([&](const auto& page) {
                    append(page,Exl3DraftHostRing::page_metadata_bytes(),
                        MetadataOwnerKind::draft_page,
                        &Exl3DraftHostRing::page_metadata_credit_belongs_to,
                        &Exl3DraftHostRing::attach_page_metadata_credit);
                });
            }
            root->state_->visit_page_metadata_owners([&](const auto& page) {
                append(page,Exl3ExactKVPage::metadata_bytes(),MetadataOwnerKind::exact_page,
                    &Exl3ExactKVPage::metadata_credit_belongs_to,
                    &Exl3ExactKVPage::attach_metadata_credit);
            });
            root->visit_tap_block_owners([&](const auto& block) {
                append(block,tap_block_metadata_bytes(),MetadataOwnerKind::tap_block,
                    &tap_block_metadata_credit_belongs_to,&attach_tap_block_metadata_credit);
            });
            root->visit_token_control_owners([&](const auto& node) {
                append(node,Exl3TokenHistory::node_metadata_bytes(),MetadataOwnerKind::token_node,
                    &Exl3TokenHistory::control_credit_belongs_to,
                    &Exl3TokenHistory::attach_control_credit);
            });
            append(root->state_,root->state_->snapshot_metadata_bytes(),
                MetadataOwnerKind::exact_state,
                &Exl3ExactHostState::snapshot_metadata_credit_belongs_to,
                &Exl3ExactHostState::attach_snapshot_metadata_credit);
        }
        // Collection completes before the first caller mutation. This also
        // keeps nested allocation failure from partially admitting metadata.
        for(const auto& allocation:allocations)visitor(allocation);
    }
    static MetadataAllocationStats metadata_allocation_stats(
        std::span<const std::shared_ptr<const Exl3VeriCacheRequest>> roots) {
        MetadataAllocationStats result;
        visit_metadata_allocations(roots,[&](const MetadataAllocation& allocation) {
            const auto kind=static_cast<std::size_t>(allocation.kind);
            if(result.unique_owners==UINT64_MAX)
                throw std::overflow_error("request metadata allocation accounting overflow");
            result.logical_bytes=checked_metadata_sum_for_test(
                result.logical_bytes,allocation.bytes);
            result.bytes_by_kind[kind]=checked_metadata_sum_for_test(
                result.bytes_by_kind[kind],allocation.bytes);
            ++result.unique_owners;
        });
        return result;
    }
    static void exercise_request_metadata_for_test() {
        const auto need=[](bool value,const char* message) {
            if(!value)throw std::runtime_error(message);
        };
        RetainedDescriptorLedger ledger;
        {
            auto request=create_planned(nullptr,1,true);
            std::array<std::vector<std::uint16_t>,5> taps;
            for(auto& plane:taps)plane.resize(5120,7);
            const auto live=bounded_shared_live_blocks_for_test<TapBlock>();bool refused=false;
            try{request->append(0,1,taps,[&](std::uint64_t bytes){return ledger.acquire(bytes-1);});}
            catch(const std::invalid_argument&){refused=true;}
            need(refused && request->blocks_.empty() && request->partitions_.empty() && ledger.bytes()==0 &&
                bounded_shared_live_blocks_for_test<TapBlock>()==live,"tap reservation refusal changed request or allocated block");
            request->append(0,1,std::move(taps),[&](std::uint64_t bytes){return ledger.acquire(bytes);});
            need(ledger.bytes()==tap_block_metadata_bytes() && request->blocks_[0]->taps[4][5119]==7,
                "credited tap append lost metadata or payload");
        }
        need(ledger.bytes()==0,"credited tap append retirement leaked");
        {
            RetainedHostAllocationLedger payload_ledger;
            auto request=create_planned(nullptr,1,true);
            std::array<std::vector<std::uint16_t>,5> taps;
            for(auto& plane:taps)plane.resize(5120,19);
            std::uint64_t payload_bytes=0;
            for(const auto& plane:taps)payload_bytes+=plane.capacity()*sizeof(std::uint16_t);
            bool refused=false;
            try{request->append(0,1,taps,{},payload_ledger.acquire(payload_bytes-1));}
            catch(const std::invalid_argument&){refused=true;}
            need(refused && payload_ledger.bytes()==0 && request->blocks_.empty(),
                "underfunded tap payload admitted or retained credit");
            ReservedHostPayloadTracker<2> limited_tracker;
            refused=false;
            try{request->append(0,1,taps,{},payload_ledger.acquire(payload_bytes),
                [&](const auto& owner,std::size_t plane,const void* data,std::size_t bytes) {
                    limited_tracker.track(owner,plane,data,bytes);
                });}catch(const std::length_error&){refused=true;}
            need(refused && payload_ledger.bytes()==0 && request->blocks_.empty() && request->partitions_.empty() &&
                limited_tracker.snapshot().allocations()==0,
                "tap tracking failure published descriptors or retained failed payload");
            request->append(0,1,std::move(taps),{},payload_ledger.acquire(payload_bytes));
            auto alias=clone(*request);
            std::uint64_t reported=0;unsigned planes=0;
            request->visit_reserved_tap_payloads(payload_ledger,[&](const auto& owner,std::size_t plane,const void* pointer,std::size_t bytes) {
                need(owner.get()==request->blocks_[0].get() && plane<5 && pointer==request->blocks_[0]->taps[plane].data(),
                    "reserved tap visitor lost allocation identity");reported+=bytes;++planes;
                need(reserved_tap_payload_matches(owner,payload_ledger,plane,pointer,bytes) &&
                    !reserved_tap_payload_matches(owner,payload_ledger,5,pointer,bytes) &&
                    !reserved_tap_payload_matches(owner,payload_ledger,plane,pointer,bytes-1) &&
                    !reserved_tap_payload_matches(owner,payload_ledger,plane,nullptr,bytes),
                    "reserved tap binding accepted changed plane or extent");
            });
            need(planes==5 && reported==payload_bytes && alias->blocks_[0]==request->blocks_[0],
                "reserved tap visitor changed shared owner or payload extent");
            RetainedHostAllocationLedger foreign;
            request->visit_reserved_tap_payloads(foreign,[&](const auto&,std::size_t,const void*,std::size_t) {
                throw std::logic_error("reserved tap visitor accepted foreign ledger");
            });
            {
                const std::array<std::shared_ptr<const Exl3VeriCacheRequest>,1> single{request};
                const std::array<std::shared_ptr<const Exl3VeriCacheRequest>,3> shared{request,alias,request};
                const auto once=reserved_tap_payload_union<5>(single,payload_ledger);
                const auto repeated=reserved_tap_payload_union<5>(shared,payload_ledger);
                need(once.allocations()==5 && repeated.allocations()==5 &&
                    once.page_bytes(4096)==repeated.page_bytes(4096),
                    "shared request tap union duplicated charged allocation pages");
                need(reserved_tap_payload_union<5>(shared,foreign).allocations()==0,
                    "request tap union admitted foreign reservation ledger");
                bool exhausted=false;
                try{(void)reserved_tap_payload_union<4>(shared,payload_ledger);}
                catch(const std::length_error&){exhausted=true;}
                need(exhausted && payload_ledger.bytes()==payload_bytes,
                    "bounded tap union refusal changed payload ownership credit");
            }
            alias.reset();
            auto retained=request->blocks_[0];std::weak_ptr<const TapBlock> weak=retained;
            request.reset();
            need(payload_ledger.bytes()==payload_bytes && retained->taps[4][5119]==19,
                "shared tap payload credit lost before last reader");
            retained.reset();
            need(weak.expired() && payload_ledger.bytes()==0,
                "raw tap payload charge survived payload destruction");
            weak.reset();
        }
        {
            auto tap=make_bounded_shared<TapBlock>();
            tap->taps[0].resize(17,23);
            const auto bytes=tap_block_metadata_bytes();
            need(!attach_tap_block_metadata_credit(tap,ledger.acquire(bytes-1)) && ledger.bytes()==0,
                "short tap metadata credit accepted or leaked");
            need(attach_tap_block_metadata_credit(tap,ledger.acquire(bytes)),"exact tap metadata credit refused");
            auto shared=tap;std::weak_ptr<TapBlock> weak=tap;tap.reset();
            need(shared->taps[0][16]==23 && ledger.bytes()==bytes &&
                tap_block_metadata_credit_belongs_to(shared,ledger),"shared tap metadata lifetime lost");
            need(!attach_tap_block_metadata_credit(shared,ledger.acquire(bytes)) && ledger.bytes()==bytes,
                "duplicate tap metadata charge retained");
            shared.reset();
            need(weak.expired() && ledger.bytes()==bytes,"tap final weak metadata released early");
            weak.reset();need(ledger.bytes()==0,"tap metadata final weak credit leaked");
        }
        {
            const auto bytes=bounded_shared_allocation_bytes<int>();
            const auto live=bounded_shared_live_blocks_for_test<int>();
            bool refused=false;
            try{auto bad=create_revision([&](std::uint64_t count){return ledger.acquire(count-1);});}
            catch(const std::invalid_argument&){refused=true;}
            need(refused && ledger.bytes()==0 && bounded_shared_live_blocks_for_test<int>()==live,
                "revision reservation refusal allocated identity or leaked charge");
            auto revision=create_revision([&](std::uint64_t count){return ledger.acquire(count);});
            auto descendant=revision;std::weak_ptr<const int> weak=revision;
            revision.reset();
            need(ledger.bytes()==bytes && !weak.expired(),"shared revision retired before descendant");
            descendant.reset();
            need(weak.expired() && ledger.bytes()==bytes,"revision final weak block charge missing");
            weak.reset();need(ledger.bytes()==0,"revision final weak credit leaked");
        }
        {
            auto bounded=create_planned(nullptr,0,true,[&](std::uint64_t bytes){return ledger.acquire(bytes);});
            const auto charged=ledger.bytes();bool refused=false;
            try {bounded->append_partition(0,1);}catch(const std::logic_error&){refused=true;}
            need(refused && bounded->partitions_.empty() && ledger.bytes()==charged,
                "credited request allowed unplanned descriptor growth");
        }
        need(ledger.bytes()==0,"empty planned request credit leaked");
        {
            RetainedDescriptorLedger revisions;
            const auto live=bounded_shared_live_blocks_for_test<Exl3VeriCacheRequest>();
            bool refused=false;
            try {auto invalid=create_planned(nullptr,8,true,
                [&](std::uint64_t bytes){return ledger.acquire(bytes);},
                [](std::uint64_t)->RetainedDescriptorLedger::Ticket{throw Exl3ResourceReservationExhausted(__FILE__,__LINE__);});}
            catch(const Exl3ResourceReservationExhausted&){refused=true;}
            need(refused && ledger.bytes()==0 && bounded_shared_live_blocks_for_test<Exl3VeriCacheRequest>()==live,
                "fresh revision refusal retained request promise or allocated request");
            auto planned=create_planned(nullptr,8,true,
                [&](std::uint64_t bytes){return ledger.acquire(bytes);},
                [&](std::uint64_t bytes){return revisions.acquire(bytes);});
            need(ledger.bytes()==planned->request_metadata_bytes() &&
                revisions.bytes()==bounded_shared_allocation_bytes<int>(),
                "planned request and fresh revision charges are not independent");
            auto shared_revision=planned->revision_;
            planned.reset();
            need(ledger.bytes()==0 && revisions.bytes()==bounded_shared_allocation_bytes<int>(),
                "request retirement released independently retained revision");
            shared_revision.reset();need(revisions.bytes()==0,"planned revision credit leaked");
        }
        {
            auto empty=create([&](std::uint64_t bytes){return ledger.acquire(bytes);});
            auto copied=clone(*empty,[&](std::uint64_t bytes){return ledger.acquire(bytes);});
            for(const auto& owner:{empty,copied}) {
                const auto charged=ledger.bytes();bool refused=false;
                try {owner->append_partition(0,1);}catch(const std::logic_error&){refused=true;}
                need(refused && owner->partitions_.empty() && ledger.bytes()==charged,
                    "credited create or clone allowed descriptor growth");
            }
        }
        need(ledger.bytes()==0,"credited create/clone capacity refusal leaked");
        {
            auto planned=create();
            const auto ring=static_cast<std::size_t>(Exl3Dflash2DraftModel::ring_capacity());
            planned->reserve_prompt_descriptors(ring*3);
            const auto* partitions=planned->partitions_.data();
            const auto* blocks=planned->blocks_.data();
            const auto partition_capacity=planned->partitions_.capacity();
            const auto block_capacity=planned->blocks_.capacity();
            // Model the worst-case one-row partition stream, including the
            // temporary insertion before retirement. No tensor work is needed.
            for(std::size_t row=0;row<ring*3;++row) {
                planned->blocks_.push_back(make_bounded_shared<TapBlock>(TapBlock{static_cast<int>(row),1,{}}));
                const auto oldest=static_cast<int>(row)+1-static_cast<int>(ring);
                while(planned->blocks_.size()>1 && planned->blocks_.front()->first+1<=oldest)
                    planned->blocks_.erase(planned->blocks_.begin());
                planned->append_partition(static_cast<int>(row),1);
                need(planned->partitions_.data()==partitions && planned->blocks_.data()==blocks &&
                    planned->partitions_.capacity()==partition_capacity && planned->blocks_.capacity()==block_capacity,
                    "request planned ring descriptors reallocated during append");
            }
            need(planned->partitions_.size()==ring && planned->blocks_.size()==ring,
                "request planned descriptors changed ring retirement");
        }
        auto parent=create();
        for(const int horizon:{2,4,8,16,32}) {
            auto inherited=create();
            inherited->append_partition(0,16);
            auto verification=create_planned(inherited.get(),static_cast<std::size_t>(horizon),true,
                [&](std::uint64_t bytes){return ledger.acquire(bytes);});
            need(request_metadata_credit_belongs_to(verification,ledger) &&
                ledger.bytes()==verification->request_metadata_bytes(),"planned verification metadata credit missing");
            const auto* storage=verification->partitions_.data();
            const auto capacity=verification->partitions_.capacity();
            // A rejection can commit any prefix; one-row appends bound the
            // descriptor requirement more tightly than wide committed blocks.
            for(int row=0;row<horizon;++row)verification->append_partition(16+row,1);
            need(verification->partitions_.data()==storage && verification->partitions_.capacity()==capacity &&
                inherited->partitions_.size()==1 && inherited->partitions_[0]==std::pair<int,int>{0,16},
                "verification descriptor plan reallocated or changed parent");
        }
        need(ledger.bytes()==0,"planned verification metadata credit leaked");
        parent->blocks_.reserve(7);parent->partitions_.reserve(11);
        parent->blocks_.push_back(make_bounded_shared<TapBlock>());
        parent->partitions_.emplace_back(0,1);
        const auto parent_bytes=parent->request_metadata_bytes();
        const auto control_bytes=bounded_shared_allocation_bytes<Exl3VeriCacheRequest>();
        const auto live_before=bounded_shared_live_blocks_for_test<Exl3VeriCacheRequest>();
        for(const int delta:{-1,1}) {
            bool refused=false;
            try {auto invalid=clone(*parent,[&](std::uint64_t bytes) {return ledger.acquire(delta<0?bytes-1:bytes+1);});}
            catch(const std::invalid_argument&) {refused=true;}
            need(refused && ledger.bytes()==0 &&
                bounded_shared_live_blocks_for_test<Exl3VeriCacheRequest>()==live_before,
                "request clone reservation extent allocated storage or leaked credit");
        }
        bool exhausted=false;
        try {auto invalid=create([](std::uint64_t)->RetainedDescriptorLedger::Ticket {throw std::bad_alloc();});}
        catch(const std::bad_alloc&) {exhausted=true;}
        need(exhausted && bounded_shared_live_blocks_for_test<Exl3VeriCacheRequest>()==live_before,
            "request exhausted reservation allocated storage");
        {
            auto reserved=clone(*parent,[&](std::uint64_t bytes) {return ledger.acquire(bytes);});
            need(request_metadata_credit_belongs_to(reserved,ledger) && ledger.bytes()==reserved->request_metadata_bytes(),
                "request preconstruction clone credit missing");
            const auto charged=ledger.bytes();const auto capacity=reserved->blocks_.capacity();
            reserved->blocks_.clear();
            need(reserved->blocks_.capacity()==capacity && ledger.bytes()==charged &&
                reserved->request_metadata_bytes()==charged,
                "compact clone released credit while descriptor capacity remains allocated");
        }
        need(ledger.bytes()==0,"request preconstruction clone retirement leaked");
        need(!attach_request_metadata_credit(parent,ledger.acquire(parent_bytes-1)) && ledger.bytes()==0,
            "request short metadata credit accepted or leaked");
        need(attach_request_metadata_credit(parent,ledger.acquire(parent_bytes)),
            "request exact metadata credit refused");
        need(!attach_request_metadata_credit(parent,ledger.acquire(parent_bytes)) && ledger.bytes()==parent_bytes,
            "request duplicate metadata credit accepted or leaked");
        auto child=clone(*parent);
        child->parent_revision_=parent->revision_;child->replay_anchor_revision_=parent->revision_;
        unsigned revision_visits=0;
        child->visit_revision_owners([&](const auto& revision){
            ++revision_visits;need(revision==parent->revision_,"revision visitor changed lineage owner");
        });
        need(revision_visits==1,"revision visitor duplicated shared lineage owner");
        need(!request_metadata_credit_belongs_to(child,ledger),"request clone inherited metadata credit");
        need(child->blocks_.data()!=parent->blocks_.data() && child->partitions_.data()!=parent->partitions_.data(),
            "request clone shares mutable descriptor buffers");
        need(child->blocks_[0]==parent->blocks_[0] && child->revision_==parent->revision_ &&
            child->partitions_==parent->partitions_,"request clone lost immutable child identity");
        const auto child_bytes=child->request_metadata_bytes();
        need(attach_request_metadata_credit(child,ledger.acquire(child_bytes)) &&
            ledger.bytes()==parent_bytes+child_bytes,"request clone independent credit failed");
        std::weak_ptr<Exl3VeriCacheRequest> parent_weak=parent,child_weak=child;
        parent.reset();
        need(parent_weak.expired() && ledger.bytes()==control_bytes+child_bytes && child->blocks_[0] &&
            child->partitions_[0]==std::pair<int,int>{0,1},"request strong retirement damaged clone or retained descriptors");
        parent_weak.reset();
        need(ledger.bytes()==child_bytes,"request final weak retirement retained parent block credit");
        child.reset();
        need(child_weak.expired() && ledger.bytes()==control_bytes,"request clone descriptor credit retirement failed");
        child_weak.reset();
        need(ledger.bytes()==0,"request clone final weak credit leaked");
    }
    template<class MakeRegistry> static void exercise_request_metadata_ceiling_for_test(MakeRegistry make_registry) {
        decltype(make_registry()) registry;
        const auto need=[](bool value,const char* message){if(!value)throw std::runtime_error(message);};
        const auto domain=static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata);
        auto exemplar=create_planned(nullptr,8,true);
        const auto bytes=exemplar->request_metadata_bytes();exemplar.reset();
        auto limits=Exl3ResourceInventory::unlimited();limits[domain]=bytes;
        registry.reset();registry=make_registry();registry->set_resource_limits(limits);
        const MetadataReservation reserve=[&](std::uint64_t count){return registry->reserve_host_metadata_lifetime(count);};
        auto owner=create_planned(nullptr,8,true,reserve);
        const auto revision=registry->revision();
        registry->admit_host_metadata_lifetime(owner,bytes,&request_metadata_credit_belongs_to,&attach_request_metadata_credit);
        need(registry->revision()==revision && registry->retained_resource_units()[domain]==bytes,
            "precredited request publication duplicated reservation");
        const auto live=bounded_shared_live_blocks_for_test<Exl3VeriCacheRequest>();
        bool refused=false;
        try{auto excess=create_planned(nullptr,8,true,reserve);}
        catch(const Exl3ResourceReservationExhausted&){refused=true;}
        need(refused && bounded_shared_live_blocks_for_test<Exl3VeriCacheRequest>()==live &&
            registry->retained_resource_units()[domain]==bytes,"request ceiling refusal allocated storage or leaked credit");
        std::weak_ptr<Exl3VeriCacheRequest> weak=owner;owner.reset();
        need(weak.expired() && registry->retained_resource_units()[domain]==bounded_shared_allocation_bytes<Exl3VeriCacheRequest>(),
            "request ceiling final strong retirement lost block or retained vectors");
        refused=false;
        try{auto excess=create_planned(nullptr,8,true,reserve);}
        catch(const Exl3ResourceReservationExhausted&){refused=true;}
        need(refused,"request ceiling ignored final weak block charge");
        weak.reset();
        auto retry=create_planned(nullptr,8,true,reserve);
        need(registry->retained_resource_units()[domain]==bytes,"request ceiling retry failed after final weak retirement");
        retry.reset();need(registry->retained_resource_units()[domain]==0,"request ceiling retry credit leaked");
        const auto revision_bytes=bounded_shared_allocation_bytes<int>();
        limits[domain]=revision_bytes;registry.reset();registry=make_registry();registry->set_resource_limits(limits);
        auto lineage=create_revision(reserve);auto descendant=lineage;
        const auto before_publication=registry->revision();
        const auto matches=[](const std::shared_ptr<const void>& value,const RetainedDescriptorLedger& ledger) noexcept {
            return bounded_split_credit_belongs_to<int>(value,ledger);
        };
        const auto attach=[](const std::shared_ptr<const void>& value,RetainedDescriptorLedger::Ticket credit) noexcept {
            return attach_bounded_split_retirement_credit<int>(value,std::move(credit),0);
        };
        registry->admit_host_metadata_lifetime(lineage,revision_bytes,matches,attach);
        registry->admit_host_metadata_lifetime(descendant,revision_bytes,matches,attach);
        need(registry->revision()==before_publication && registry->retained_resource_units()[domain]==revision_bytes,
            "shared revision publication charged duplicate metadata");
        const auto revision_blocks=bounded_shared_live_blocks_for_test<int>();
        refused=false;try{auto excess=create_revision(reserve);}
        catch(const Exl3ResourceReservationExhausted&){refused=true;}
        need(refused && bounded_shared_live_blocks_for_test<int>()==revision_blocks,
            "revision ceiling refusal allocated identity");
        std::weak_ptr<const int> lineage_weak=lineage;lineage.reset();
        need(!lineage_weak.expired() && registry->retained_resource_units()[domain]==revision_bytes,
            "revision ceiling lost descendant ownership charge");
        descendant.reset();
        need(lineage_weak.expired() && registry->retained_resource_units()[domain]==revision_bytes,
            "revision ceiling released final weak storage early");
        lineage_weak.reset();
        auto revision_retry=create_revision(reserve);revision_retry.reset();
        need(registry->retained_resource_units()[domain]==0,"revision ceiling retry leaked metadata");
        const auto combined_bytes=bytes+revision_bytes;
        for(const auto ceiling:{bytes,combined_bytes-1}) {
            limits[domain]=ceiling;registry.reset();registry=make_registry();registry->set_resource_limits(limits);
            const auto requests_before=bounded_shared_live_blocks_for_test<Exl3VeriCacheRequest>();
            const auto revisions_before=bounded_shared_live_blocks_for_test<int>();
            refused=false;
            try{auto excess=create_planned(nullptr,8,true,reserve,reserve);}
            catch(const Exl3ResourceReservationExhausted&){refused=true;}
            need(refused && registry->retained_resource_units()[domain]==0 &&
                bounded_shared_live_blocks_for_test<Exl3VeriCacheRequest>()==requests_before &&
                bounded_shared_live_blocks_for_test<int>()==revisions_before,
                "combined ceiling refusal allocated request/revision or retained provisional credit");
        }
        limits[domain]=combined_bytes;registry.reset();registry=make_registry();registry->set_resource_limits(limits);
        auto combined=create_planned(nullptr,8,true,reserve,reserve);
        const auto combined_revision=registry->revision();
        registry->admit_host_metadata_lifetime(combined,bytes,&request_metadata_credit_belongs_to,&attach_request_metadata_credit);
        combined->visit_revision_owners([&](const auto& identity){
            registry->admit_host_metadata_lifetime(identity,revision_bytes,matches,attach);
        });
        need(registry->revision()==combined_revision && registry->retained_resource_units()[domain]==combined_bytes,
            "combined precredited publication duplicated request/revision charges");
        std::weak_ptr<Exl3VeriCacheRequest> combined_weak=combined;
        std::weak_ptr<const int> combined_identity_weak=combined->revision_;
        combined.reset();
        need(combined_weak.expired() && combined_identity_weak.expired() &&
            registry->retained_resource_units()[domain]==bounded_shared_allocation_bytes<Exl3VeriCacheRequest>()+revision_bytes,
            "combined strong retirement lost weak storage or retained descriptor credit");
        combined_weak.reset();
        need(registry->retained_resource_units()[domain]==revision_bytes,"combined request weak release altered revision credit");
        combined_identity_weak.reset();
        need(registry->retained_resource_units()[domain]==0,"combined final weak retirement leaked");
        // Isolate shared tap metadata; these fixture request shells are uncredited.
        const auto tap_bytes=tap_block_metadata_bytes();
        limits[domain]=tap_bytes;registry.reset();registry=make_registry();registry->set_resource_limits(limits);
        auto tap_parent=create_planned(nullptr,2,true);
        std::array<std::vector<std::uint16_t>,5> payload;
        for(auto& plane:payload)plane.resize(5120,31);
        for(const int first:{-1,std::numeric_limits<int>::max()}) {
            bool invalid=false;
            try{tap_parent->append(first,1,payload,reserve);}catch(const std::invalid_argument&){invalid=true;}
            need(invalid && tap_parent->blocks_.empty() && tap_parent->partitions_.empty(),
                "invalid tap position changed authoritative descriptors");
            invalid=false;
            try{tap_parent->append_partition(first,1);}catch(const std::invalid_argument&){invalid=true;}
            need(invalid && tap_parent->partitions_.empty(),"invalid partition position admitted");
        }
        tap_parent->append(0,1,payload,reserve);
        auto tap_child=clone(*tap_parent);
        const auto tap_revision=registry->revision();
        for(const auto& root:{tap_parent,tap_child})root->visit_tap_block_owners([&](const auto& block){
            registry->admit_host_metadata_lifetime(block,tap_bytes,&tap_block_metadata_credit_belongs_to,
                &attach_tap_block_metadata_credit);
        });
        need(registry->revision()==tap_revision && registry->retained_resource_units()[domain]==tap_bytes,
            "shared tap publication duplicated metadata reservation");
        const auto tap_blocks=bounded_shared_live_blocks_for_test<TapBlock>();
        refused=false;try{tap_parent->append(1,1,payload,reserve);}
        catch(const Exl3ResourceReservationExhausted&){refused=true;}
        need(refused && bounded_shared_live_blocks_for_test<TapBlock>()==tap_blocks &&
            tap_parent->blocks_.size()==1 && tap_parent->partitions_.size()==1,
            "tap ceiling refusal allocated block or changed authoritative partitions");
        std::weak_ptr<const TapBlock> tap_weak=tap_parent->blocks_[0];
        tap_parent->blocks_.clear();
        need(!tap_weak.expired() && tap_child->blocks_[0]->taps[4][5119]==31 &&
            registry->retained_resource_units()[domain]==tap_bytes,"compaction released shared tap charge or payload");
        tap_child->blocks_.clear();
        need(tap_weak.expired() && registry->retained_resource_units()[domain]==tap_bytes,
            "tap descriptor clear released final weak block charge");
        tap_weak.reset();
        tap_parent->append(1,1,std::move(payload),reserve);
        need(registry->retained_resource_units()[domain]==tap_bytes,"tap ceiling did not recover after final weak release");
        tap_parent.reset();tap_child.reset();
        need(registry->retained_resource_units()[domain]==0,"shared tap ceiling fixture retained metadata");
    }
    // Full-page attachment is authorized by the acquired immutable root, not by
    // equal offsets or token counts in a context callback. Typed MRoPE roots do
    // not enter the ordinary sequential-text page route.
    bool owns_shared_text_page(const std::shared_ptr<const Exl3ExactKVPage>& page,
        const std::shared_ptr<const void>& model,int rope_offset) const noexcept {
        if(prepared_identity_ || !state_ || !page || !page.use_count() ||
           !model || !model.use_count() || state_->rope_offset()!=rope_offset)return false;
        const auto expected_model=state_->model_identity();
        if(model.get()!=expected_model.get() || model.owner_before(expected_model) ||
           expected_model.owner_before(model))return false;
        const auto expected=state_->shared_kv_page(page->first);
        return expected && expected.get()==page.get() &&
            !expected.owner_before(page) && !page.owner_before(expected);
    }
    bool same_input_identity(const Exl3VeriCacheRequest& other) const {
        return prepared_identity_ && other.prepared_identity_ ?
            prepared_identity_->equals(*other.prepared_identity_) :
            !prepared_identity_ && !other.prepared_identity_;
    }
    // Research-only native encoder -> exact represented-media L2 execution.
    // No constructor accepts arbitrary embeddings beside caller-supplied identity.
    static std::shared_ptr<const Exl3VeriCacheRequest> initialize_prepared_numeric(
        Exl3TextContext& exact,Exl3VisionContext& vision,
        const targets::qwen3_6::PreparedPrompt& prompt,
        const std::function<bool()>& cancelled={},
        const Exl3EncodedMediaRetentionReserve& reserve={});
    static std::shared_ptr<const Exl3VeriCacheRequest> replay_prepared_numeric(
        Exl3TextContext& exact,Exl3VisionContext& vision,
        const Exl3PreparedMediaReplay& replay,
        const std::function<bool()>& cancelled={},
        const Exl3EncodedMediaRetentionReserve& reserve={});
    bool is_child_of(const Exl3VeriCacheRequest& root) const noexcept {
        return parent_revision_==root.revision_;
    }
    Exl3CommittedTapBinding committed_tap_binding(
        const std::shared_ptr<Exl3TextContext>& context,
        std::uint64_t acquisition,std::uint64_t execution) const {
        const auto context_model=context?context->model_identity():std::shared_ptr<const void>{};
        const auto state_model=state_?state_->model_identity():std::shared_ptr<const void>{};
        const bool same_model=context_model && state_model &&
            context_model.get()==state_model.get() &&
            !context_model.owner_before(state_model) && !state_model.owner_before(context_model);
        if(!context || !state_ || !acquisition || !execution || !same_model ||
           context->position()!=state_->position())
            throw std::invalid_argument("committed tap root binding scope mismatch");
        return {context,revision_,state_->position(),acquisition,execution};
    }
    bool owns_committed_tap_root(const Exl3CommittedTapBinding& binding) const noexcept {
        return binding.root_revision && state_ &&
            binding.root_revision.get()==revision_.get() &&
            !binding.root_revision.owner_before(revision_) &&
            !revision_.owner_before(binding.root_revision) &&
            binding.root_position==state_->position();
    }
    // Collapse two already verified private transitions for one coordinator
    // publication. Preserve the final state/history; prove both revision edges.
    std::shared_ptr<const Exl3VeriCacheRequest> compose_verified_child(
        const Exl3VeriCacheRequest& parent,const Exl3VeriCacheRequest& intermediate,
        const Exl3TextContext::SnapshotMetadataReservation& reserve={}) const {
        if(!is_child_of(intermediate) || !intermediate.is_child_of(parent) ||
            !state_ || !intermediate.state_ || !parent.state_ ||
            state_->model_identity()!=parent.state_->model_identity() ||
            intermediate.state_->position()<=parent.state_->position() ||
            state_->position()<=intermediate.state_->position())
            throw std::invalid_argument("composed verified child lineage");
        auto result=Exl3VeriCacheRequest::clone(*this,reserve);
        result->revision_=create_revision(reserve);
        result->parent_revision_=parent.revision_;
        return result;
    }
    const std::shared_ptr<const Exl3ExactHostState>& state() const noexcept {return state_;}
    bool compact_draft() const noexcept {return projected_!=nullptr;}
    const std::shared_ptr<const Exl3DraftHostRing>& projected_metadata_owner() const noexcept {return projected_;}
    std::shared_ptr<const Exl3DraftHostRing> detached_projected_conditioning_for_test() const {
        return projected_?projected_->detached_payload_for_test():nullptr;
    }
    bool matches_detached_conditioning_for_test(const Exl3DraftHostRing& copy) const {
        return projected_ && projected_->same_represented_payload_for_test(copy);
    }
    bool same_projected_conditioning_for_test(const Exl3VeriCacheRequest& other) const {
        return projected_ && other.projected_ && projected_->same_payload(*other.projected_);
    }
    std::vector<std::int64_t> token_suffix(int first=0) const {return history_.suffix(first);}
    std::size_t token_count() const noexcept {return static_cast<std::size_t>(history_.position());}
    std::uint64_t token_fingerprint() const noexcept {return history_.fingerprint();}
    bool matches_tokens(std::span<const std::int64_t> tokens) const noexcept {return history_.equals(tokens);}
    bool same_tokens(const Exl3VeriCacheRequest& other) const noexcept {return history_.same_tokens(other.history_);}
    Exl3VeriCacheReplay replay_plan(std::shared_ptr<const Exl3VeriCacheRequest> anchor={}) const;
    static Exl3TokenHistory::Stats token_storage_stats(std::span<const std::shared_ptr<const Exl3VeriCacheRequest>> requests) {
        std::vector<const Exl3TokenHistory*> histories;
        for(const auto& request:requests) {
            if(!request) throw std::invalid_argument("token accounting null request");
            histories.push_back(&request->history_);
        }
        return Exl3TokenHistory::visit(histories);
    }
    std::size_t tap_bytes() const noexcept {
        std::size_t bytes=0;
        for(const auto& block:blocks_) for(const auto& tap:block->taps) bytes+=tap.size()*2;
        return bytes;
    }
    static std::size_t allocated_tap_bytes(std::span<const std::shared_ptr<const Exl3VeriCacheRequest>> requests) {
        std::unordered_set<const TapBlock*> seen;
        std::size_t bytes=0;
        for(const auto& request:requests) if(request) for(const auto& block:request->blocks_)
            if(seen.insert(block.get()).second) for(const auto& tap:block->taps) bytes+=tap.capacity()*2;
        return bytes;
    }
    static std::uint64_t allocated_projected_bytes(std::span<const std::shared_ptr<const Exl3VeriCacheRequest>> requests) {
        std::vector<std::shared_ptr<const Exl3DraftHostRing>> rings;
        for(const auto& request:requests) if(request && request->projected_) rings.push_back(request->projected_);
        return Exl3DraftHostRing::visit_allocations(rings);
    }
    static void visit_host_allocations(std::span<const std::shared_ptr<const Exl3VeriCacheRequest>> requests,
        const std::function<void(const void*,std::size_t)>& visitor,bool include_unused_capacity=true) {
        if(!visitor) throw std::invalid_argument("request allocation visitor missing");
        for(const auto& request:requests)
            if(!request)throw std::invalid_argument("request allocation null image");
        std::vector<std::shared_ptr<const Exl3ExactHostState>> images;
        std::vector<const Exl3TokenHistory*> histories;
        std::vector<std::shared_ptr<const Exl3DraftHostRing>> rings;
        // Every request contributes one image/history and at most one ring.
        // Establish these traversal extents before invoking payload visitors.
        images.reserve(requests.size());
        histories.reserve(requests.size());
        rings.reserve(requests.size());
        std::unordered_set<const TapBlock*> seen;
        std::unordered_set<const Exl3PreparedIdentity*> seen_identities;
        for(const auto& request:requests) {
            images.push_back(request->state_);
            histories.push_back(&request->history_);
            if(request->prepared_identity_ && seen_identities.insert(request->prepared_identity_.get()).second)
                request->prepared_identity_->visit_allocations(visitor,include_unused_capacity);
            if(request->projected_) rings.push_back(request->projected_);
            for(const auto& block:request->blocks_) if(seen.insert(block.get()).second)
                for(const auto& plane:block->taps) {
                    const auto elements=include_unused_capacity?plane.capacity():plane.size();
                    if(elements) visitor(plane.data(),elements*2);
                }
        }
        Exl3ExactHostState::visit_host_allocations(images,visitor,include_unused_capacity);
        Exl3TokenHistory::visit(histories,visitor,include_unused_capacity);
        Exl3DraftHostRing::visit_allocations(rings,visitor,include_unused_capacity);
    }

    // Exact token-state equality, independent of forward call partitions.
    bool same_taps(const Exl3VeriCacheRequest& other) const {
        if(projected_ || other.projected_) throw std::logic_error("compact requests retain projected conditioning, not full target tap payloads");
        if(state_->position()!=other.state_->position()) return false;
        const int first=std::max(0,state_->position()-Exl3Dflash2DraftModel::ring_keep());
        for(int layer=0;layer<5;++layer) {
            std::vector<std::uint16_t> a,b;
            const auto flatten=[&](const auto& blocks,auto& output) {
                for(const auto& block:blocks) {
                    const int skip=std::clamp(first-block->first,0,block->rows);
                    output.insert(output.end(),block->taps[layer].begin()+skip*5120,block->taps[layer].end());
                }
            };
            flatten(blocks_,a);flatten(other.blocks_,b);
            if(a!=b) return false;
        }
        return true;
    }

    // Ordinary full-reference initialization, with every actual target tap
    // captured before the next forward overwrites it. No approximate upstream.
    static std::shared_ptr<const Exl3VeriCacheRequest> initialize(
        Exl3TextContext& exact,std::span<const std::int64_t> prompt,int prefill_rows=8,
        const std::function<void(int)>& progress={},bool actual_tail=false) {
        validate_prefill_width(prefill_rows);
        if(prompt.empty() || prompt.size()>static_cast<std::size_t>(exact.max_context()) || exact.continuation_capacity()<8)
            throw std::invalid_argument("request initialization needs prompt and prepared8");
        for(auto token:prompt) if(token<0 || token>=248320) throw std::invalid_argument("request prompt token extent");
        auto result=Exl3VeriCacheRequest::create_planned(nullptr,prompt.size(),true,exact.request_metadata_reservation(),exact.request_metadata_reservation());
        exact.reset();
        const int initial=static_cast<int>(std::min<std::size_t>(16,prompt.size()));
        exact.prefill(prompt.first(initial));
        result->append(0,initial,exact.exact_tap_rows_host(),exact.request_metadata_reservation());
        if(progress) progress(exact.position());
        result->ingest_prompt_suffix(exact,prompt.subspan(initial),prefill_rows,progress,{},actual_tail);
        result->state_=exact.export_exact_host_state();
        result->history_=Exl3TokenHistory{}.append(prompt,exact.request_metadata_reservation());
        return result;
    }

    // Terminal boundaries for a request whose target and draft stayed device
    // resident through its generated windows. The caller has already completed
    // both streams and exported one matching pair of immutable host snapshots.
    // These factories never execute or re-label speculative model work.
    static std::shared_ptr<const Exl3VeriCacheRequest> initialize_device_root(
        Exl3TextContext& exact,Exl3Dflash2DraftModel& draft,
        std::span<const std::int64_t> prompt,cudaStream_t stream=nullptr) {
        if(prompt.empty() || prompt.size()>static_cast<std::size_t>(exact.max_context()) ||
           exact.position()!=static_cast<int>(prompt.size()) ||
           draft.ring_base_abs()+draft.ring_count()!=exact.position())
            throw std::invalid_argument("device root target/draft/prompt frontier");
        for(const auto token:prompt)
            if(token<0 || token>=248320)
                throw std::invalid_argument("device root prompt token extent");
        const auto& reserve=exact.request_metadata_reservation();
        auto target_state=exact.export_exact_host_state(stream);
        auto draft_ring=draft.export_host_ring(stream,true,reserve);
        if(draft_ring->position()!=target_state->position())
            throw std::logic_error("device root snapshot frontier changed");
        auto result=create_planned(nullptr,0,false,reserve,reserve);
        result->history_=Exl3TokenHistory{}.append(prompt,reserve);
        result->state_=std::move(target_state);
        result->projected_=std::move(draft_ring);
        return result;
    }

    std::shared_ptr<const Exl3VeriCacheRequest> append_device_terminal(
        Exl3TextContext& exact,Exl3Dflash2DraftModel& draft,
        std::span<const std::int64_t> committed_suffix,
        cudaStream_t stream=nullptr) const {
        if(!state_ || !projected_ || prepared_identity_ ||
           committed_suffix.empty() ||
           exact.model_identity()!=state_->model_identity() ||
           exact.position()!=state_->position()+
               static_cast<int>(committed_suffix.size()) ||
           draft.ring_base_abs()+draft.ring_count()!=exact.position())
            throw std::invalid_argument("device terminal target/draft lineage");
        for(const auto token:committed_suffix)
            if(token<0 || token>=248320)
                throw std::invalid_argument("device terminal token extent");
        const auto& reserve=exact.request_metadata_reservation();
        auto target_state=exact.export_exact_host_state(stream);
        auto draft_ring=draft.export_host_ring(stream,true,reserve);
        if(draft_ring->position()!=target_state->position())
            throw std::logic_error("device terminal snapshot frontier changed");
        auto result=create_planned(this,0,false,reserve,reserve);
        result->parent_revision_=revision_;
        result->replay_anchor_revision_=
            replay_anchor_revision_?replay_anchor_revision_:revision_;
        result->history_=history_.append(committed_suffix,reserve);
        result->state_=std::move(target_state);
        result->projected_=std::move(draft_ring);
        return result;
    }

    // Device-route prompt extension: restores this root (target exact state and
    // draft ring), lets `ingest` execute the suffix on the target and commit its
    // taps to the draft, then snapshots the extended root.
    std::shared_ptr<const Exl3VeriCacheRequest> append_device_prompt(
        Exl3TextContext& exact,Exl3Dflash2DraftModel& draft,
        std::span<const std::int64_t> suffix,const std::function<void()>& ingest,
        cudaStream_t stream=nullptr,bool resident=false) const {
        if(!state_ || !projected_ || prepared_identity_ || suffix.empty() || !ingest ||
           exact.model_identity()!=state_->model_identity() ||
           suffix.size()>static_cast<std::size_t>(exact.max_context()-state_->position()))
            throw std::invalid_argument("device prompt root/suffix extent");
        for(const auto token:suffix)
            if(token<0 || token>=248320)
                throw std::invalid_argument("device prompt token extent");
        // `resident`: the caller's context and drafter still hold exactly this root.
        if(resident) {
            if(exact.position()!=state_->position() ||
               draft.ring_base_abs()+draft.ring_count()!=exact.position())
                throw std::logic_error("device prompt resident root frontier");
        } else {
            exact.restore_exact_host_state(*state_,stream);
            restore_draft(draft,{},stream);
        }
        ingest();
        if(exact.position()!=state_->position()+static_cast<int>(suffix.size()) ||
           draft.ring_base_abs()+draft.ring_count()!=exact.position())
            throw std::logic_error("device prompt ingestion frontier");
        const auto& reserve=exact.request_metadata_reservation();
        auto target_state=exact.export_exact_host_state(stream);
        auto draft_ring=draft.export_host_ring(stream,true,reserve);
        auto result=create_planned(this,0,false,reserve,reserve);
        result->parent_revision_=revision_;
        result->replay_anchor_revision_=
            replay_anchor_revision_?replay_anchor_revision_:revision_;
        result->history_=history_.append(suffix,reserve);
        result->state_=std::move(target_state);
        result->projected_=std::move(draft_ring);
        return result;
    }

    // User-supplied prompt suffix, distinct from speculative token publication.
    // Forks the exact root, preserving only proven shared-prefix ownership.
    std::shared_ptr<const Exl3VeriCacheRequest> append_prompt(
        Exl3TextContext& exact,std::span<const std::int64_t> suffix,int prefill_rows=8,
        const std::function<void(int)>& progress={}) const {
        if(projected_) throw std::logic_error("compact prompt extension requires an explicit draft streaming path");
        if(prepared_identity_) throw std::logic_error("prepared root needs typed control suffix execution");
        validate_prefill_width(prefill_rows);
        if(suffix.empty() || exact.continuation_capacity()<8 || state_->position()>exact.max_context() ||
            suffix.size()>static_cast<std::size_t>(exact.max_context()-state_->position()))
            throw std::invalid_argument("request prompt suffix extent");
        for(auto token:suffix) if(token<0 || token>=248320) throw std::invalid_argument("request suffix token extent");
        auto result=Exl3VeriCacheRequest::create_planned(this,suffix.size(),true,exact.request_metadata_reservation(),exact.request_metadata_reservation());
        result->parent_revision_=revision_;
        result->replay_anchor_revision_=replay_anchor_revision_?replay_anchor_revision_:revision_;
        // Reserve the immutable token path before changing the physical context.
        result->history_=history_.append(suffix,exact.request_metadata_reservation());
        exact.restore_exact_host_state(*state_);
        try {
            result->ingest_prompt_suffix(exact,suffix,prefill_rows,progress);
            result->state_=exact.export_exact_host_state();
        } catch(...) {
            const auto failure=std::current_exception();
            try{exact.restore_exact_host_state(*state_);}catch(...){}
            std::rethrow_exception(failure);
        }
        return result;
    }

    // A new user/control suffix extends a complete hybrid root. Stream only
    // freshly authoritative tap partitions; retained projected history is never
    // relabeled as full taps or reconstructed from tentative draft features.
    std::shared_ptr<const Exl3VeriCacheRequest> append_prompt_compact(
        Exl3TextContext& exact,Exl3Dflash2DraftModel& draft,
        const std::array<std::uint16_t*,5>& staging,std::span<const std::int64_t> suffix,
        int prefill_rows=1024,const std::function<void(int)>& progress={}) const {
        if(prepared_identity_) throw std::logic_error("prepared root needs typed control suffix execution");
        if(!projected_)return append_prompt(exact,suffix,prefill_rows,progress);
        validate_prefill_width(prefill_rows);
        if(suffix.empty() || exact.continuation_capacity()<8 || state_->position()>exact.max_context() ||
            suffix.size()>static_cast<std::size_t>(exact.max_context()-state_->position()))
            throw std::invalid_argument("compact prompt suffix extent");
        for(auto p:staging)if(!p)throw std::invalid_argument("compact prompt staging missing");
        for(auto token:suffix)if(token<0 || token>=248320)throw std::invalid_argument("compact prompt token extent");
        auto result=Exl3VeriCacheRequest::create_planned(this,suffix.size(),false,exact.request_metadata_reservation(),exact.request_metadata_reservation());
        result->parent_revision_=revision_;
        result->replay_anchor_revision_=replay_anchor_revision_?replay_anchor_revision_:revision_;
        result->history_=history_.append(suffix,exact.request_metadata_reservation());
        exact.restore_exact_host_state(*state_);restore_draft(draft,staging);
        try {
            result->ingest_prompt_suffix(exact,suffix,prefill_rows,progress,
                [&](int first,int rows,const auto& taps){
                    std::array<const std::uint16_t*,5> pointers{};
                    for(int tap=0;tap<5;++tap){
                        const auto error=cudaMemcpy(staging[tap],taps[tap].data(),rows*5120ULL*2,cudaMemcpyHostToDevice);
                        if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
                        pointers[tap]=staging[tap];
                    }
                    draft.commit_prefill_block(pointers.data(),rows,first);
                });
            result->state_=exact.export_exact_host_state();result->projected_=draft.export_host_ring(nullptr,true,exact.request_metadata_reservation());
            result->blocks_.clear();
            return result;
        }catch(...){
            const auto failure=std::current_exception();
            try{exact.restore_exact_host_state(*state_);}catch(...){}
            try{restore_draft(draft,staging);}catch(...){}
            std::rethrow_exception(failure);
        }
    }

    std::pair<std::shared_ptr<const Exl3VeriCacheRequest>,Exl3OuterReferenceResult> verify(
        Exl3TextContext& exact,std::span<const std::int64_t> tentative,
        std::span<const std::int64_t> terminal={},cudaStream_t stream=nullptr) const {
        if(projected_) throw std::logic_error("compact request needs verify_compact with its resident drafter");
        auto result=Exl3VeriCacheRequest::create_planned(this,tentative.size(),true,exact.request_metadata_reservation(),exact.request_metadata_reservation());
        result->parent_revision_=revision_;
        result->replay_anchor_revision_=replay_anchor_revision_;
        auto verified=tentative.size()==1 ? verify_exl3_outer_reference(
                exact,*state_,tentative,true,terminal,stream,true) :
            tentative.size()<=8 ? verify_exl3_outer_batched_reference(
                exact,*state_,tentative,true,terminal,stream,true) :
            verify_exl3_outer_windowed_reference(
                exact,*state_,tentative,true,terminal,stream,true);
        result->append(state_->position(),static_cast<int>(verified.committed_tokens.size()),verified.committed_taps,
            exact.request_metadata_reservation());
        result->state_=verified.committed_state;
        result->history_=history_.append(verified.committed_tokens,exact.request_metadata_reservation());
        if(prepared_identity_) result->prepared_identity_=prepared_identity_->append_text(verified.committed_tokens.size(),exact.request_metadata_reservation());
        return {std::move(result),std::move(verified)};
    }

    // Explicit generated/control text at the current typed MRoPE frontier.
    // Incoming rendered user prompts still require typed matching/execution.
    std::shared_ptr<const Exl3VeriCacheRequest> append_control_compact(
        Exl3TextContext& exact,Exl3Dflash2DraftModel& draft,
        const std::array<std::uint16_t*,5>& staging,std::span<const std::int64_t> tokens) const {
        if(!prepared_identity_) return append_prompt_compact(exact,draft,staging,tokens,8);
        auto next_identity=prepared_identity_->append_text(tokens.size(),exact.request_metadata_reservation());
        auto execution_parent=Exl3VeriCacheRequest::clone(*this,exact.request_metadata_reservation());
        execution_parent->prepared_identity_.reset();
        auto executed=execution_parent->append_prompt_compact(exact,draft,staging,tokens,8);
        auto result=Exl3VeriCacheRequest::clone(*executed,exact.request_metadata_reservation());
        result->prepared_identity_=std::move(next_identity);
        return result;
    }

    std::shared_ptr<const Exl3VeriCacheRequest> compact_draft(
        Exl3Dflash2DraftModel& draft,const std::array<std::uint16_t*,5>& staging,
        const Exl3TextContext::SnapshotMetadataReservation& reserve={}) const {
        auto result=Exl3VeriCacheRequest::clone(*this,reserve);
        if(!projected_) {
            restore_draft(draft,staging);
            result->projected_=draft.export_host_ring(nullptr,true,reserve);
            result->blocks_.clear();
        }
        return result;
    }

    std::pair<std::shared_ptr<const Exl3VeriCacheRequest>,Exl3OuterReferenceResult> verify_compact(
        Exl3TextContext& exact,Exl3Dflash2DraftModel& draft,const std::array<std::uint16_t*,5>& staging,
        std::span<const std::int64_t> tentative,std::span<const std::int64_t> terminal={},cudaStream_t stream=nullptr,
        bool checkpoint_repair=false,Exl3CommittedTapBinding tap_binding={},
        unsigned tap_copy_fault_for_test=0) const {
        if(!projected_ || exact.model_identity()!=state_->model_identity())
            throw std::invalid_argument("compact verification reference mismatch");
        const char* direct_requested=std::getenv("NINFER_EXL3_COMPACT_DIRECT_TAP_STAGING");
        if(direct_requested && std::string_view(direct_requested)!="0" &&
            std::string_view(direct_requested)!="1")
            throw std::invalid_argument("compact direct tap staging must be 0 or 1");
        const bool direct_staging=direct_requested && std::string_view(direct_requested)=="1";
        const auto* device_requested=std::getenv("NINFER_EXL3_COMMITTED_TAP_D2D");
        if(device_requested && std::string_view(device_requested)!="0" && std::string_view(device_requested)!="1")
            throw std::invalid_argument("committed tap D2D must be 0 or 1");
        const auto* device_only_requested=std::getenv("NINFER_EXL3_COMPACT_DEVICE_TAPS_ONLY");
        if(device_only_requested && std::string_view(device_only_requested)!="0" &&
            std::string_view(device_only_requested)!="1")
            throw std::invalid_argument("compact device taps only must be 0 or 1");
        const bool device_taps_only=device_only_requested &&
            std::string_view(device_only_requested)=="1";
        // One bounded draft-ingestion partition. Native8+native8 remains two
        // target segments with independent source metadata, not an M16 target
        // computation; both may assemble into the existing private 16-row stage.
        const bool device_staging=device_requested && std::string_view(device_requested)=="1" &&
            tentative.size()<=16 && static_cast<bool>(tap_binding);
        if(device_taps_only && (!direct_staging || !device_staging))
            throw std::invalid_argument(
                "compact device taps only requires direct D2D tap staging");
        if(device_staging && !owns_committed_tap_root(tap_binding))
            throw std::invalid_argument("committed tap root ancestry mismatch");
        if(tap_copy_fault_for_test && (!device_staging || tap_copy_fault_for_test>80))
            throw std::invalid_argument("tap copy fault requires active D2D copy1..80");
        for(auto pointer:staging) if(!pointer) throw std::invalid_argument("compact draft staging missing");
        const auto* witness=std::getenv("NINFER_EXL3_RESIDENT_DRAFT_RING");
        if(witness && std::string_view(witness)!="0" && std::string_view(witness)!="1")
            throw std::invalid_argument("resident draft ring must be 0 or 1");
        auto result=Exl3VeriCacheRequest::create_planned(this,tentative.size(),false,exact.request_metadata_reservation(),exact.request_metadata_reservation());
        result->parent_revision_=revision_;result->replay_anchor_revision_=replay_anchor_revision_;
        if(witness && std::string_view(witness)=="1") draft.restore_host_ring_if_needed(projected_,stream);
        else draft.restore_host_ring(projected_,stream);
        // Keep the actual async-copy source alive through failure rollback.
        Exl3OuterReferenceResult verified;
        std::size_t staged_tap_bytes=0;
        unsigned tap_copy_index=0;
        Exl3CommittedTapConsumer tap_consumer;
        if(device_staging)tap_consumer=[&](const Exl3CommittedTapSegment& segment,
                                           cudaStream_t copy_stream) {
            segment.validate();
            if(segment.owner!=tap_binding.context_owner ||
               !owns_committed_tap_root({tap_binding.context_owner,segment.root_revision,
                   segment.root_position,segment.acquisition,segment.execution}) ||
               segment.model_identity!=exact.model_identity() ||
               segment.acquisition!=tap_binding.acquisition ||
               segment.execution!=tap_binding.execution)
                throw std::invalid_argument("committed tap D2D source scope mismatch");
            for(std::size_t plane=0;plane<staging.size();++plane) {
                const auto error=++tap_copy_index==tap_copy_fault_for_test?
                    cudaErrorUnknown:cudaMemcpyAsync(
                    staging[plane]+static_cast<std::size_t>(segment.destination_first)*5120,
                    segment.planes[plane]+static_cast<std::size_t>(segment.source_first)*5120,
                    static_cast<std::size_t>(segment.rows)*5120*2,
                    cudaMemcpyDeviceToDevice,copy_stream);
                if(error!=cudaSuccess)throw std::runtime_error(
                    tap_copy_fault_for_test?"injected committed tap D2D copy failure":
                    cudaGetErrorString(error));
            }
            staged_tap_bytes+=static_cast<std::size_t>(segment.rows)*5*5120*2;
        };
        const auto* binding=device_staging?&tap_binding:nullptr;
        const auto* consumer=device_staging?&tap_consumer:nullptr;
        const bool capture_host_taps=!device_taps_only;
        try {
            verified=tentative.size()==1?verify_exl3_outer_reference(
                    exact,*state_,tentative,capture_host_taps,terminal,stream,true,binding,consumer):
                checkpoint_repair && tentative.size()<=8?
                    verify_exl3_outer_checkpointed_reference(
                    exact,*state_,tentative,capture_host_taps,terminal,stream,true,binding,consumer):
                tentative.size()<=8?verify_exl3_outer_batched_reference(
                    exact,*state_,tentative,capture_host_taps,terminal,stream,true,binding,consumer):
                verify_exl3_outer_windowed_reference(
                    exact,*state_,tentative,capture_host_taps,terminal,stream,true,binding,consumer);
            verified.committed_tap_d2d_bytes=staged_tap_bytes;
            const int rows=static_cast<int>(verified.committed_tokens.size());
            if(device_taps_only) {
                if(std::any_of(verified.committed_taps.begin(),verified.committed_taps.end(),
                    [](const auto& tap){return !tap.empty();}))
                    throw std::logic_error("device-only committed taps reached host");
                verified.committed_tap_host_export_rows_avoided=rows;
            } else for(const auto& tap:verified.committed_taps)
                if(tap.size()!=static_cast<std::size_t>(rows)*5120)
                    throw std::logic_error("compact committed host tap extent");
            std::array<const std::uint16_t*,5> pointers{};
            for(int offset=0;offset<rows;offset+=16) {
                const int block_rows=std::min(16,rows-offset);
                std::optional<RetainedHostAllocationLedger::Ticket> payload_credit;
                if(!direct_staging && exact.request_host_payload_reservation()) {
                    const auto bytes=5ULL*block_rows*5120*sizeof(std::uint16_t);
                    payload_credit.emplace(exact.request_host_payload_reservation()(bytes));
                    if(payload_credit->bytes()!=bytes)throw std::invalid_argument("compact tap payload reservation extent");
                }
                std::array<std::vector<std::uint16_t>,5> block;
                for(int tap=0;tap<5;++tap) {
                    if(!direct_staging) {
                        const auto begin=verified.committed_taps[tap].begin()+
                            static_cast<std::size_t>(offset)*5120;
                        block[tap].assign(begin,begin+static_cast<std::size_t>(block_rows)*5120);
                    }
                    if(!device_staging) {
                        const auto* source=verified.committed_taps[tap].data()+
                            static_cast<std::size_t>(offset)*5120;
                        const auto error=cudaMemcpyAsync(staging[tap],source,
                            static_cast<std::size_t>(block_rows)*5120*2,cudaMemcpyHostToDevice,stream);
                        if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
                    }
                    pointers[tap]=staging[tap];
                }
                if(direct_staging) result->append_partition(state_->position()+offset,block_rows);
                else result->append(state_->position()+offset,block_rows,std::move(block),exact.request_metadata_reservation(),
                    std::move(payload_credit),exact.request_host_payload_observer());
                draft.commit_prefill_block(pointers.data(),block_rows,state_->position()+offset,stream);
            }
            result->projected_=draft.export_host_ring(stream,true,exact.request_metadata_reservation());result->blocks_.clear();
            if(result->projected_->position()!=verified.committed_state->position())
                throw std::logic_error("compact committed draft position mismatch");
            result->state_=verified.committed_state;result->history_=history_.append(verified.committed_tokens,exact.request_metadata_reservation());
            if(prepared_identity_) result->prepared_identity_=prepared_identity_->append_text(verified.committed_tokens.size(),exact.request_metadata_reservation());
            return {std::move(result),std::move(verified)};
        } catch(...) {
            const auto failure=std::current_exception();
            try{exact.restore_exact_host_state(*state_,stream);}catch(...){}
            try{draft.restore_host_ring(projected_,stream);}catch(...){}
            std::rethrow_exception(failure);
        }
    }

    // Caller owns five GPU staging buffers of at least16*5120 FP16 elements.
    // Replays original projection call partitions and absolute positions. This
    // reconstructs only the bounded draft ring, never the full target prefix.
    bool draft_lineage_ready(const Exl3Dflash2DraftModel& draft,
        std::uint64_t acquisition,std::uint64_t execution,int position) const noexcept {
        return projected_ && state_ && position==state_->position() &&
            projected_->position()==position && draft.host_ring_lineage(projected_,acquisition,execution);
    }
    bool rebind_draft_scope_if_resident(Exl3Dflash2DraftModel& draft,
        std::uint64_t acquisition,std::uint64_t execution) const {
        return projected_ && state_ && projected_->position()==state_->position() &&
            draft.rebind_ring_scope_if_resident(projected_,acquisition,execution);
    }
    // True means physical restore/reconstruction work was required. Compact
    // roots may return false only from the exact resident-ring witness.
    bool restore_draft_if_needed(Exl3Dflash2DraftModel& draft,
        const std::array<std::uint16_t*,5>& staging,cudaStream_t stream=nullptr) const {
        if(projected_) {
            if(projected_->position()!=state_->position()) throw std::logic_error("compact draft target position mismatch");
            return draft.restore_host_ring_if_needed(projected_,stream);
        }
        restore_draft(draft,staging,stream);return true;
    }
    void restore_draft(Exl3Dflash2DraftModel& draft,
        const std::array<std::uint16_t*,5>& staging,cudaStream_t stream=nullptr) const {
        if(projected_) {
            if(projected_->position()!=state_->position()) throw std::logic_error("compact draft target position mismatch");
            draft.restore_host_ring(projected_,stream);
            return;
        }
        for(auto pointer:staging) if(!pointer) throw std::invalid_argument("draft staging missing");
        const auto check=[](cudaError_t error) {
            if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        };
        draft.reset(stream);
        for(const auto& block:blocks_) {
            std::array<const std::uint16_t*,5> pointers{};
            for(int tap=0;tap<5;++tap) {
                check(cudaMemcpyAsync(staging[tap],block->taps[tap].data(),block->taps[tap].size()*2,
                    cudaMemcpyHostToDevice,stream));
                pointers[tap]=staging[tap];
            }
            draft.commit_prefill_block(pointers.data(),block->rows,block->first,stream);
        }
        check(cudaStreamSynchronize(stream));
        if(draft.ring_base_abs()+draft.ring_count()!=state_->position())
            throw std::runtime_error("restored draft ownership mismatch");
    }

    void restore_inner(Exl3TextContext& inner,Exl3Dflash2DraftModel& draft,
        const std::array<std::uint16_t*,5>& staging,
        const Exl3TurboAngleWarmPages* warm=nullptr,cudaStream_t stream=nullptr) const {
        inner.restore_oscar_host_state(*state_,warm,stream);
        restore_draft(draft,staging,stream);
    }
};

// Bounded in-process lookup for already-authoritative request roots. Hashes are
// accelerators only: every hit verifies the resident model identity, the exact
// caller-supplied compatibility contract, and every token. The compatibility
// contract is deliberately opaque to this layer so the engine entry point can
// bind tokenizer/artifact/configuration/mask/modality identity without teaching
// the numerical request object about serving policy. Returned shared ownership
// remains valid if another publisher subsequently replaces or evicts the slot.
class Exl3VeriCachePrefixIndex {
    struct Entry;
public:
    static constexpr std::size_t maximum_entries=64;
    using HashFunction=std::function<std::uint64_t(
        std::span<const std::int64_t>,std::string_view)>;
    struct StorageStats {
        std::size_t entries=0;
        std::uint64_t payload_allocated_bytes=0;
        std::uint64_t identity_allocated_bytes=0;
        std::uint64_t accounted_bytes=0;
        // Inline owner storage, not evictable entry data. Already included when
        // the enclosing cache object is inventoried by sizeof; do not add twice.
        std::uint64_t inline_lookup_metadata_bytes=maximum_entries*sizeof(std::size_t);
    };
    struct Admission {
        bool admitted=false,replaced=false;
        std::uint64_t generation=0; // Set only for a committed entry.
        std::size_t evicted=0;
        std::uint64_t configured_budget_bytes=0,effective_budget_bytes=0;
        StorageStats storage;
    };
    struct RetentionMetadata {
        std::shared_ptr<const Exl3VeriCacheRequest> root;
        std::uint64_t generation=0;
        std::size_t reusable_tokens=0;
        // Unique within this root, not exclusive to it or additive across roots.
        std::uint64_t retained_payload_bytes=0;
        std::optional<std::uint64_t> preparation_microseconds;
        std::optional<std::uint64_t> observed_accesses;
        std::uint64_t lookup_selections=0; // Selections, not completed reuse or saved work.
    };
    std::vector<RetentionMetadata> retention_metadata() const {
        std::shared_lock lock(mutex_);
        std::vector<RetentionMetadata> result;result.reserve(entries_.size());
        for(const auto& entry:entries_) {
            const std::array<const Entry*,1> one{&entry};
            result.push_back({entry.root,entry.generation,entry.root->token_count(),
                account(one).payload_allocated_bytes,entry.preparation_microseconds,entry.observed_accesses,
                entry.lookup_selections.value.load(std::memory_order_relaxed)});
        }
        return result;
    }
    bool supply_retention_observation(const std::shared_ptr<const Exl3VeriCacheRequest>& root,
        std::uint64_t generation,std::optional<std::uint64_t> preparation_microseconds,
        std::optional<std::uint64_t> observed_accesses) {
        if(!root || !generation)return false;
        std::unique_lock lock(mutex_);
        for(auto& entry:entries_)if(entry.root==root && entry.generation==generation) {
            entry.preparation_microseconds=preparation_microseconds;
            entry.observed_accesses=observed_accesses;return true;
        }
        return false;
    }

    explicit Exl3VeriCachePrefixIndex(std::size_t capacity,HashFunction hasher={})
        : capacity_(capacity),hasher_(std::move(hasher)) {
        if(!capacity_) throw std::invalid_argument("prefix index capacity must be positive");
        if(capacity_>maximum_entries)
            throw std::invalid_argument("prefix index capacity exceeds bounded range index");
    }
    Exl3VeriCachePrefixIndex(const Exl3VeriCachePrefixIndex&)=delete;
    Exl3VeriCachePrefixIndex& operator=(const Exl3VeriCachePrefixIndex&)=delete;

    std::size_t size() const noexcept {
        std::shared_lock lock(mutex_);return entries_.size();
    }
    std::size_t capacity() const noexcept {return capacity_;}
    void exhaust_generations_for_test() {
        std::unique_lock lock(mutex_);next_generation_=std::numeric_limits<std::uint64_t>::max();
    }
    void fail_next_publication_growth_for_test() {
        std::unique_lock lock(mutex_);
        if(fail_publication_growth_)throw std::logic_error("prefix growth failure already armed");
        fail_publication_growth_=true;
    }
    void fail_next_policy_commit_for_test() {
        std::unique_lock lock(mutex_);
        if(fail_policy_commit_)throw std::logic_error("prefix policy failure already armed");
        fail_policy_commit_=true;
    }
    std::size_t identity_allocated_bytes() const noexcept {
        std::shared_lock lock(mutex_);std::size_t bytes=0;
        for(const auto& entry:entries_)
            bytes+=entry.contract.capacity();
        return bytes;
    }
    StorageStats storage_stats() const {
        std::shared_lock lock(mutex_);
        std::vector<const Entry*> entries;entries.reserve(entries_.size());
        for(const auto& entry:entries_) entries.push_back(&entry);
        return account(entries);
    }

    void publish(std::shared_ptr<const Exl3VeriCacheRequest> root,
        std::string compatibility_contract) {
        if(!root || !root->state()) throw std::invalid_argument("prefix index root missing");
        if(compatibility_contract.empty()) throw std::invalid_argument("prefix index compatibility contract missing");
        const auto count=root->token_count();
        if(!count || count!=static_cast<std::size_t>(root->state()->position()))
            throw std::invalid_argument("prefix index token/state extent mismatch");
        const auto hash=hasher_?hasher_(root->token_suffix(),compatibility_contract):hash_contract(root->token_fingerprint(),compatibility_contract);
        const auto model=root->state()->model_identity();
        std::unique_lock lock(mutex_);
        // Obtain any new vector slot before erasing a replacement or FIFO root.
        // Entry moves below transfer only already-owned buffers and shared roots.
        if(entries_.size()<capacity_ && entries_.capacity()==entries_.size()) {
            if(fail_publication_growth_) {fail_publication_growth_=false;throw std::bad_alloc();}
            const auto remaining=capacity_-entries_.size();
            const auto growth=std::min(remaining,std::max<std::size_t>(1,entries_.size()));
            entries_.reserve(entries_.size()+growth);
        }
        const auto generation=take_generation();
        for(auto it=entries_.begin();it!=entries_.end();++it) {
            if(it->hash==hash && it->model_identity==model &&
                it->contract==compatibility_contract && it->root->same_tokens(*root) && it->root->same_input_identity(*root)) {
                Entry replacement{hash,generation,std::move(model),
                    std::move(compatibility_contract),std::move(root)};
                entries_.erase(it);entries_.push_back(std::move(replacement));rebuild_lengths();return;
            }
        }
        if(entries_.size()==capacity_) entries_.erase(entries_.begin());
        entries_.push_back(Entry{hash,generation,std::move(model),
            std::move(compatibility_contract),std::move(root)});
        rebuild_lengths();
    }

    std::shared_ptr<const Exl3VeriCacheRequest> lookup(
        const Exl3TextContext& exact,std::span<const std::int64_t> tokens,
        std::string_view compatibility_contract,const Exl3PreparedIdentity* prepared=nullptr) const {
        if(tokens.empty() || compatibility_contract.empty()) return {};
        const auto hash=hasher_?hasher_(tokens,compatibility_contract):stable_hash(tokens,compatibility_contract);
        const auto model=exact.model_identity();
        std::shared_lock lock(mutex_);
        // Newest exact publication wins; collisions and stale predecessors are
        // harmless because the complete identity is checked before return.
        for(auto it=entries_.rbegin();it!=entries_.rend();++it)
            if(it->hash==hash && it->model_identity==model &&
                it->contract==compatibility_contract &&
                it->root->token_count()==tokens.size() &&
                matches_prepared(*it,prepared,false) &&
                it->root->matches_tokens(tokens.first(it->root->token_count())))
                return selected_root(*it);
        return {};
    }

    // Serving callers may expose several explicit cacheable frontiers. Select
    // the longest resident exact prefix, breaking equal-length ties by newest
    // publication. Token and compatibility comparisons remain authoritative.
    std::shared_ptr<const Exl3VeriCacheRequest> lookup_longest(
        const Exl3TextContext& exact,std::span<const std::int64_t> tokens,
        std::string_view compatibility_contract,std::size_t minimum_tokens=1,
        const Exl3PreparedIdentity* prepared=nullptr) const {
        if(tokens.empty() || compatibility_contract.empty() || minimum_tokens>tokens.size()) return {};
        const auto model=exact.model_identity();
        std::shared_lock lock(mutex_);
        const auto end=length_order_.begin()+entries_.size();
        auto candidate=std::lower_bound(length_order_.begin(),end,tokens.size(),
            [&](std::size_t index,std::size_t size){return entries_[index].root->token_count()>size;});
        for(;candidate!=end;++candidate) {
            const auto& entry=entries_[*candidate];
            if(entry.root->token_count()<minimum_tokens)break;
            if(entry.model_identity==model && entry.contract==compatibility_contract &&
                matches_prepared(entry,prepared,true) &&
                entry.root->matches_tokens(tokens.first(entry.root->token_count())))return selected_root(entry);
        }
        return {};
    }

    // Typed media prefixes use the complete position proof rather than the
    // older whole-prompt rope-delta equality. The resident root's own state
    // remains authoritative for its frontier offset.
    std::shared_ptr<const Exl3VeriCacheRequest> lookup_longest_prepared_positions(
        const Exl3TextContext& exact,std::span<const std::int64_t> tokens,
        std::string_view compatibility_contract,const Exl3PreparedIdentity& incoming,
        std::size_t minimum_tokens=1) const {
        if(tokens.empty() || compatibility_contract.empty() || minimum_tokens>tokens.size())return {};
        const auto model=exact.model_identity();std::shared_lock lock(mutex_);
        const auto end=length_order_.begin()+entries_.size();
        auto candidate=std::lower_bound(length_order_.begin(),end,tokens.size(),
            [&](std::size_t index,std::size_t size){return entries_[index].root->token_count()>size;});
        for(;candidate!=end;++candidate) {
            const auto& entry=entries_[*candidate];const auto count=entry.root->token_count();
            const auto* resident=entry.root->prepared_identity();
            if(count<minimum_tokens)break;
            if(entry.model_identity==model && entry.contract==compatibility_contract && resident &&
               entry.root->state() && entry.root->state()->rope_offset()==resident->rope_delta() &&
               resident->prefix_positions_equal(incoming,count) &&
               entry.root->matches_tokens(tokens.first(count)))return selected_root(entry);
        }
        return {};
    }

    struct RetentionDecision {
        std::shared_ptr<const Exl3VeriCacheRequest> root;
        std::uint64_t generation=0;
        bool required=true;
        // Caller-supplied fixed-point value per byte, in one common scale for
        // this transaction. No inference from token counts or lookup selections.
        std::optional<std::uint64_t> value_per_byte;
    };
    // Add one immutable root under both the existing entry bound and a cache-
    // owned byte bound. The physical cap is supplied by the serving owner from
    // its existing memory gate; this layer never invents availability. An
    // individually oversize candidate leaves the current cache unchanged.
    Admission admit(std::shared_ptr<const Exl3VeriCacheRequest> root,
        std::string compatibility_contract,std::uint64_t byte_budget,
        std::uint64_t available_physical_bytes,std::uint64_t physical_reserve_bytes,
        std::optional<std::span<const RetentionDecision>> retention=std::nullopt) {
        if(!byte_budget) throw std::invalid_argument("prefix index byte budget must be positive");
        Admission result;result.configured_budget_bytes=byte_budget;
        result.effective_budget_bytes=available_physical_bytes>physical_reserve_bytes?
            std::min(byte_budget,available_physical_bytes-physical_reserve_bytes):0;
        std::unique_lock lock(mutex_);
        if(retention) {
            if(entries_.size()>64 || retention->size()!=entries_.size()) {
                result.storage=current_stats_locked();return result;
            }
            for(std::size_t i=0;i<entries_.size();++i)
                if((*retention)[i].root!=entries_[i].root ||
                    (*retention)[i].generation!=entries_[i].generation) {
                    result.storage=current_stats_locked();return result;
                }
        }
        Entry candidate=make_entry(std::move(root),std::move(compatibility_contract));
        std::vector<const Entry*> prospective;prospective.reserve(entries_.size()+1);
        for(std::size_t i=0;i<entries_.size();++i) {
            if(!same_identity(entries_[i],candidate)) {
                prospective.push_back(&entries_[i]);
            } else {
                // A representation renewal (for example the Engine's compact root replacing
                // the freshly prepared root) is still the same exact input identity. Carry its
                // measured preparation/access observations and selection count forward so the
                // final admitted authority does not erase the cold-work evidence.
                candidate.preparation_microseconds=entries_[i].preparation_microseconds;
                candidate.observed_accesses=entries_[i].observed_accesses;
                candidate.lookup_selections=entries_[i].lookup_selections;
                if(retention && (*retention)[i].required) {
                    result.storage=current_stats_locked();return result;
                }
            }
        }
        prospective.push_back(&candidate);
        const std::array<const Entry*,1> candidate_only{&candidate};
        const auto candidate_stats=account(candidate_only);
        if(candidate_stats.accounted_bytes>result.effective_budget_bytes) {
            result.storage=current_stats_locked();return result;
        }
        if(retention) {
            auto policy_stats=account(prospective);
            while(prospective.size()>capacity_ || policy_stats.accounted_bytes>result.effective_budget_bytes) {
                const auto decision=[&](const Entry* entry)->const RetentionDecision& {
                    return (*retention)[static_cast<std::size_t>(entry-entries_.data())];
                };
                bool measured=true;
                for(const auto* entry:prospective)if(entry!=&candidate && !decision(entry).required &&
                    !decision(entry).value_per_byte)measured=false;
                auto victim=prospective.end();
                for(auto it=prospective.begin();it!=prospective.end();++it) {
                    if(*it==&candidate || decision(*it).required)continue;
                    if(victim==prospective.end() || (measured &&
                        *decision(*it).value_per_byte<*decision(*victim).value_per_byte))victim=it;
                }
                if(victim==prospective.end()) {result.storage=current_stats_locked();return result;}
                prospective.erase(victim);policy_stats=account(prospective);
            }
        }
        std::size_t first=prospective.size()>capacity_?prospective.size()-capacity_:0;
        auto selected=[&] {return std::span<const Entry* const>(prospective).subspan(first);};
        auto stats=account(selected());
        while(stats.accounted_bytes>result.effective_budget_bytes && first+1<prospective.size()) {
            ++first;stats=account(selected());
        }
        if(retention && fail_policy_commit_) {fail_policy_commit_=false;throw std::bad_alloc();}
        std::vector<Entry> next;next.reserve(selected().size());
        result.replaced=std::any_of(entries_.begin(),entries_.end(),
            [&](const Entry& entry){return same_identity(entry,candidate);});
        std::size_t kept_old=0;
        auto found=entries_.begin();
        for(const Entry* wanted:selected()) if(wanted!=&candidate) {
            // prospective/selected preserve publication order. Never rescan
            // previously consumed entries when moving the retained suffix.
            while(found!=entries_.end() && &*found!=wanted)++found;
            if(found==entries_.end()) throw std::logic_error("prefix index admission selection drift");
            next.push_back(std::move(*found));++found;++kept_old;
        }
        result.evicted=entries_.size()-kept_old-(result.replaced?1:0);
        result.generation=candidate.generation;
        next.push_back(std::move(candidate));entries_=std::move(next);rebuild_lengths();
        // Selected roots and contract buffers were moved, not rebuilt. Reuse the
        // exact precommit accounting instead of allocating after cache mutation.
        result.admitted=true;result.storage=stats;return result;
    }

    StorageStats trim_to_budget(std::uint64_t byte_budget,
        std::uint64_t available_physical_bytes,std::uint64_t physical_reserve_bytes,
        std::size_t* evicted=nullptr) {
        if(!byte_budget) throw std::invalid_argument("prefix index byte budget must be positive");
        const auto effective=available_physical_bytes>physical_reserve_bytes?
            std::min(byte_budget,available_physical_bytes-physical_reserve_bytes):0;
        std::unique_lock lock(mutex_);std::size_t count=0;auto stats=current_stats_locked();
        while(!entries_.empty() && stats.accounted_bytes>effective) {
            entries_.erase(entries_.begin());rebuild_lengths();++count;stats=current_stats_locked();
        }
        if(evicted) *evicted=count;return stats;
    }

    struct PolicyTrim {
        bool inputs_current=false,budget_met=false;
        std::size_t evicted=0;
        StorageStats storage;
    };
    // Optional, bounded cache-retention transaction. The owner must serialize
    // changes to required coverage with this call. Dropping a cache reference
    // neither unlocks publication coverage nor proves physical bytes reclaimed;
    // active readers retain their own strong roots and allocation ownership.
    PolicyTrim trim_with_retention_policy(std::uint64_t byte_budget,
        std::uint64_t available_physical_bytes,std::uint64_t physical_reserve_bytes,
        std::span<const RetentionDecision> decisions) {
        if(!byte_budget)throw std::invalid_argument("prefix policy byte budget must be positive");
        std::unique_lock lock(mutex_);
        PolicyTrim result;result.storage=current_stats_locked();
        const auto effective=available_physical_bytes>physical_reserve_bytes?
            std::min(byte_budget,available_physical_bytes-physical_reserve_bytes):0;
        result.budget_met=result.storage.accounted_bytes<=effective;
        if(entries_.size()>64 || decisions.size()!=entries_.size())return result;
        // Require the complete ordered generation snapshot; no stale priority
        // or permission may silently attach to a replacement or alias entry.
        for(std::size_t i=0;i<entries_.size();++i)
            if(decisions[i].root!=entries_[i].root ||
                decisions[i].generation!=entries_[i].generation)return result;
        result.inputs_current=true;
        std::array<bool,64> removed{};
        std::vector<const Entry*> selected;selected.reserve(entries_.size());
        while(!result.budget_met) {
            bool measured=true;
            for(std::size_t i=0;i<entries_.size();++i)
                if(!removed[i] && !decisions[i].required && !decisions[i].value_per_byte)
                    measured=false;
            std::size_t victim=entries_.size();
            for(std::size_t i=0;i<entries_.size();++i) {
                if(removed[i] || decisions[i].required)continue;
                if(victim==entries_.size() || (measured &&
                    *decisions[i].value_per_byte<*decisions[victim].value_per_byte))victim=i;
            }
            if(victim==entries_.size())break;
            removed[victim]=true;++result.evicted;
            selected.clear();
            for(std::size_t i=0;i<entries_.size();++i)if(!removed[i])selected.push_back(&entries_[i]);
            result.storage=account(selected);
            result.budget_met=result.storage.accounted_bytes<=effective;
        }
        // All allocating accounting and destination growth precede mutation.
        if(result.evicted) {
            if(fail_policy_commit_) {fail_policy_commit_=false;throw std::bad_alloc();}
            std::vector<Entry> next;next.reserve(entries_.size()-result.evicted);
            for(std::size_t i=0;i<entries_.size();++i)
                if(!removed[i])next.push_back(std::move(entries_[i]));
            entries_=std::move(next);rebuild_lengths();
        }
        return result;
    }

    std::vector<std::shared_ptr<const Exl3VeriCacheRequest>> roots() const {
        std::shared_lock lock(mutex_);
        std::vector<std::shared_ptr<const Exl3VeriCacheRequest>> result;result.reserve(entries_.size());
        for(const auto& entry:entries_) result.push_back(entry.root);
        return result;
    }
    void copy_roots_into(std::vector<std::shared_ptr<const Exl3VeriCacheRequest>>& output) const {
        std::shared_lock lock(mutex_);
        if(output.capacity()<entries_.size())
            throw std::length_error("prefix root snapshot exceeds reserved capacity");
        output.clear();
        for(const auto& entry:entries_)output.push_back(entry.root);
    }

    bool erase(const Exl3TextContext& exact,std::span<const std::int64_t> tokens,
        std::string_view compatibility_contract,const Exl3PreparedIdentity* prepared=nullptr) {
        if(tokens.empty() || compatibility_contract.empty()) return false;
        const auto hash=hasher_?hasher_(tokens,compatibility_contract):stable_hash(tokens,compatibility_contract);
        const auto model=exact.model_identity();
        std::unique_lock lock(mutex_);
        for(auto it=entries_.begin();it!=entries_.end();++it)
            if(it->hash==hash && it->model_identity==model &&
                it->contract==compatibility_contract &&
                it->root->token_count()==tokens.size() &&
                matches_prepared(*it,prepared,false) &&
                it->root->matches_tokens(tokens.first(it->root->token_count()))) {
                entries_.erase(it);rebuild_lengths();return true;
            }
        return false;
    }

    void clear() noexcept {
        std::unique_lock lock(mutex_);entries_.clear();
    }

private:
    struct SelectionCount {
        mutable std::atomic<std::uint64_t> value{0};
        SelectionCount()=default;
        SelectionCount(const SelectionCount& other) noexcept:value(other.value.load(std::memory_order_relaxed)) {}
        SelectionCount& operator=(const SelectionCount& other) noexcept {
            value.store(other.value.load(std::memory_order_relaxed),std::memory_order_relaxed);return *this;
        }
    };
    struct Entry {
        std::uint64_t hash=0,generation=0;
        std::shared_ptr<const void> model_identity;
        std::string contract;

        std::shared_ptr<const Exl3VeriCacheRequest> root;
        std::optional<std::uint64_t> preparation_microseconds,observed_accesses;
        SelectionCount lookup_selections;
    };
    static std::shared_ptr<const Exl3VeriCacheRequest> selected_root(const Entry& entry) {
        auto count=entry.lookup_selections.value.load(std::memory_order_relaxed);
        while(count!=std::numeric_limits<std::uint64_t>::max() &&
            !entry.lookup_selections.value.compare_exchange_weak(count,count+1,std::memory_order_relaxed)) {}
        return entry.root;
    }
    static bool same_identity(const Entry& left,const Entry& right) {
        return left.hash==right.hash && left.model_identity==right.model_identity &&
            left.contract==right.contract && left.root->same_tokens(*right.root) &&
            left.root->same_input_identity(*right.root);
    }
    static bool matches_prepared(const Entry& entry,const Exl3PreparedIdentity* incoming,bool prefix) {
        const auto* resident=entry.root->prepared_identity();
        if(!resident || !incoming) return !resident && !incoming;
        return prefix?resident->prefix_equals(*incoming,entry.root->token_count()):resident->equals(*incoming);
    }
    Entry make_entry(std::shared_ptr<const Exl3VeriCacheRequest> root,
        std::string compatibility_contract) {
        if(!root || !root->state()) throw std::invalid_argument("prefix index root missing");
        if(compatibility_contract.empty()) throw std::invalid_argument("prefix index compatibility contract missing");
        const auto count=root->token_count();
        if(!count || count!=static_cast<std::size_t>(root->state()->position()))
            throw std::invalid_argument("prefix index token/state extent mismatch");
        const auto hash=hasher_?hasher_(root->token_suffix(),compatibility_contract):hash_contract(root->token_fingerprint(),compatibility_contract);
        return Entry{hash,take_generation(),root->state()->model_identity(),
            std::move(compatibility_contract),std::move(root)};
    }
    static StorageStats account(std::span<const Entry* const> entries) {
        StorageStats result;result.entries=entries.size();
        std::vector<std::shared_ptr<const Exl3VeriCacheRequest>> roots;roots.reserve(entries.size());
        for(const auto* entry:entries) {
            if(!entry) throw std::invalid_argument("prefix index accounting null entry");
            roots.push_back(entry->root);
            const auto identity=static_cast<std::uint64_t>(entry->contract.capacity());
            if(identity>std::numeric_limits<std::uint64_t>::max()-result.identity_allocated_bytes)
                throw std::overflow_error("prefix index identity accounting overflow");
            result.identity_allocated_bytes+=identity;
        }
        if(!roots.empty()) Exl3VeriCacheRequest::visit_host_allocations(roots,
            [&](const void*,std::size_t bytes) {
                if(bytes>std::numeric_limits<std::uint64_t>::max()-result.payload_allocated_bytes)
                    throw std::overflow_error("prefix index payload accounting overflow");
                result.payload_allocated_bytes+=bytes;
            });
        if(result.identity_allocated_bytes>std::numeric_limits<std::uint64_t>::max()-result.payload_allocated_bytes)
            throw std::overflow_error("prefix index total accounting overflow");
        result.accounted_bytes=result.payload_allocated_bytes+result.identity_allocated_bytes;
        return result;
    }
    StorageStats current_stats_locked() const {
        std::vector<const Entry*> entries;entries.reserve(entries_.size());
        for(const auto& entry:entries_) entries.push_back(&entry);
        return account(entries);
    }
    static std::uint64_t hash_contract(std::uint64_t value,std::string_view contract) noexcept {
        const auto byte=[&](std::uint8_t input) {value^=input;value*=1099511628211ULL;};
        for(char c:contract) byte(static_cast<std::uint8_t>(c));
        byte(0xff);
        return value;
    }
    static std::uint64_t stable_hash(std::span<const std::int64_t> tokens,std::string_view contract) noexcept {
        return hash_contract(Exl3TokenHistory::extend_fingerprint(1469598103934665603ULL,tokens),contract);
    }

    const std::size_t capacity_;
    HashFunction hasher_;
    mutable std::shared_mutex mutex_;
    std::vector<Entry> entries_;
    bool fail_publication_growth_=false;
    bool fail_policy_commit_=false;
    // Fixed metadata: no allocation or borrowed Entry pointers during lookup.
    // Index positions are rebuilt under the mutation lock after every erase/move.
    std::array<std::size_t,maximum_entries> length_order_{};
    void rebuild_lengths() noexcept {
        for(std::size_t i=0;i<entries_.size();++i)length_order_[i]=i;
        std::sort(length_order_.begin(),length_order_.begin()+entries_.size(),
            [&](std::size_t a,std::size_t b) {
                const auto left=entries_[a].root->token_count(),right=entries_[b].root->token_count();
                return left!=right?left>right:a>b;
            });
    }
    std::uint64_t next_generation_=1;
    std::uint64_t take_generation() {
        if(next_generation_==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("prefix index generation exhausted");
        return next_generation_++;
    }
};

// Structured serving identity encoded into the opaque T45 compatibility
// contract. Length framing avoids delimiter aliases. Model-instance identity
// remains independently enforced by Exl3VeriCachePrefixIndex.
struct Exl3VeriCacheServingIdentity {
    std::string process_namespace,artifact_identity,tokenizer_identity,
        configuration_identity,modality_identity;
    std::string contract() const {
        std::string result="ninfer-exl3-serving-prefix-v1";
        const auto append=[&](std::string_view name,const std::string& value) {
            if(value.empty() || value.size()>(1U<<20))
                throw std::invalid_argument("serving prefix identity component extent");
            result.push_back('|');result.append(name);result.push_back('=');
            result+=std::to_string(value.size());result.push_back(':');result+=value;
        };
        append("namespace",process_namespace);append("artifact",artifact_identity);
        append("tokenizer",tokenizer_identity);append("configuration",configuration_identity);
        append("modality",modality_identity);return result;
    }
};

// Minimal single-lane serving owner for the already-qualified immutable exact
// root/index/attach machinery. Cacheability is an explicit frontend decision;
// this class never guesses a semantic prefix boundary.
class Exl3VeriCacheServingPrefixCache {
public:
    struct Policy {
        std::size_t entry_capacity=1,minimum_prefix_tokens=64;
        std::uint64_t byte_budget=0,physical_reserve_bytes=0;
    };
    struct Metrics {
        std::size_t prompt_tokens=0,reused_prompt_tokens=0,executed_prompt_tokens=0;
        double lookup_ms=0,prefill_ms=0,attach_ms=0;
        bool cache_hit=false,admitted=false,replaced=false;
        bool preparation_reused=false;
        std::size_t evicted=0;
        std::uint64_t configured_budget_bytes=0,effective_budget_bytes=0;
        Exl3VeriCachePrefixIndex::StorageStats cache_storage;
    };
    struct Result {
        std::shared_ptr<const Exl3VeriCacheRequest> request;
        Metrics metrics;
    };

    Exl3VeriCacheServingPrefixCache(Policy policy,Exl3VeriCacheServingIdentity identity)
        : policy_(policy),contract_(identity.contract()),index_(policy.entry_capacity) {
        if(!policy_.entry_capacity || !policy_.minimum_prefix_tokens || !policy_.byte_budget ||
            !policy_.physical_reserve_bytes)
            throw std::invalid_argument("serving prefix cache policy extent");
    }
    Exl3VeriCacheServingPrefixCache(const Exl3VeriCacheServingPrefixCache&)=delete;
    Exl3VeriCacheServingPrefixCache& operator=(const Exl3VeriCacheServingPrefixCache&)=delete;

    Result prepare(Exl3TextContext& exact,std::span<const std::int64_t> prompt,
        std::size_t cacheable_prefix_tokens,std::uint64_t available_physical_bytes,
        int prefill_rows=1024,const std::function<void(int)>& progress={},
        Exl3Dflash2DraftModel* draft=nullptr,const std::array<std::uint16_t*,5>* staging=nullptr,
        bool reuse_complete_input=false) {
        if(prompt.empty() || cacheable_prefix_tokens>prompt.size() ||
            (cacheable_prefix_tokens && cacheable_prefix_tokens<policy_.minimum_prefix_tokens))
            throw std::invalid_argument("serving cacheable prefix extent");
        std::unique_lock service_lock(service_mutex_);
        const auto extend=[&](const auto& root,std::span<const std::int64_t> suffix) {
            if(!root->compact_draft())return root->append_prompt(exact,suffix,prefill_rows,progress);
            if(!draft || !staging)throw std::invalid_argument("compact cached prefix needs resident drafter");
            return root->append_prompt_compact(exact,*draft,*staging,suffix,prefill_rows,progress);
        };
        Result output;auto& metrics=output.metrics;metrics.prompt_tokens=prompt.size();
        metrics.configured_budget_bytes=policy_.byte_budget;
        std::shared_ptr<const Exl3VeriCacheRequest> prefix;
        const auto lookup_start=std::chrono::steady_clock::now();
        if(cacheable_prefix_tokens) prefix=index_.lookup_longest(exact,
            reuse_complete_input?prompt:prompt.first(cacheable_prefix_tokens),contract_,policy_.minimum_prefix_tokens);
        if(prefix && reuse_complete_input)
            cacheable_prefix_tokens=std::max(cacheable_prefix_tokens,static_cast<std::size_t>(prefix->state()->position()));
        metrics.lookup_ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-lookup_start).count();
        metrics.cache_hit=static_cast<bool>(prefix);
        metrics.reused_prompt_tokens=prefix?static_cast<std::size_t>(prefix->state()->position()):0;
        metrics.executed_prompt_tokens=prompt.size()-metrics.reused_prompt_tokens;
        const auto prefill_start=std::chrono::steady_clock::now();
        if(cacheable_prefix_tokens) {
            if(!prefix) prefix=Exl3VeriCacheRequest::initialize(exact,
                prompt.first(cacheable_prefix_tokens),prefill_rows,progress);
            else if(static_cast<std::size_t>(prefix->state()->position())<cacheable_prefix_tokens)
                prefix=extend(prefix,prompt.subspan(prefix->state()->position(),
                    cacheable_prefix_tokens-prefix->state()->position()));
            else exact.restore_exact_host_state(*prefix->state());
            if(!metrics.preparation_reused && static_cast<std::size_t>(prefix->state()->position())==cacheable_prefix_tokens &&
                (!metrics.cache_hit || metrics.reused_prompt_tokens<cacheable_prefix_tokens)) {
                const auto cold_preparation_us=std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now()-prefill_start).count();
                const auto admission=index_.admit(prefix,contract_,policy_.byte_budget,
                    available_physical_bytes,policy_.physical_reserve_bytes);
                if(admission.admitted && !metrics.cache_hit && cold_preparation_us>=0)
                    index_.supply_retention_observation(prefix,admission.generation,
                        static_cast<std::uint64_t>(cold_preparation_us),std::nullopt);
                metrics.admitted=admission.admitted;metrics.replaced=admission.replaced;
                metrics.evicted=admission.evicted;metrics.effective_budget_bytes=admission.effective_budget_bytes;
                metrics.cache_storage=admission.storage;
            }
        }
        const auto attach_start=std::chrono::steady_clock::now();
        if(!cacheable_prefix_tokens) output.request=Exl3VeriCacheRequest::initialize(
            exact,prompt,prefill_rows,progress);
        else if(cacheable_prefix_tokens<prompt.size()) output.request=extend(
            prefix,prompt.subspan(cacheable_prefix_tokens));
        else output.request=prefix;
        metrics.attach_ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-attach_start).count();
        metrics.prefill_ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-prefill_start).count();
        if(!metrics.effective_budget_bytes) metrics.effective_budget_bytes=
            available_physical_bytes>policy_.physical_reserve_bytes?
                std::min(policy_.byte_budget,available_physical_bytes-policy_.physical_reserve_bytes):0;
        if(!metrics.cache_storage.entries) metrics.cache_storage=index_.storage_stats();
        return output;
    }

    Result prepare_media(Exl3TextContext& exact,Exl3VisionContext& vision,
        const targets::qwen3_6::PreparedPrompt& prompt,bool cacheable_input,
        std::uint64_t available_physical_bytes,const std::function<bool()>& cancelled={},
        const Exl3EncodedMediaRetentionReserve& reserve={});

    // Explicit independent-context preparation path. Cache lookup and the
    // numerical suffix remain outside the service lock; reload identity is
    // snapshotted and checked before any admission and again before return.
    Result prepare_concurrent(Exl3TextContext& exact,
        std::span<const std::int64_t> prompt,
        std::size_t cacheable_prefix_tokens,
        std::uint64_t available_physical_bytes,int prefill_rows=1024,
        const std::function<void(int)>& progress={},
        Exl3Dflash2DraftModel* draft=nullptr,const std::array<std::uint16_t*,5>* staging=nullptr,
        bool reuse_complete_input=false) {
        // Reject impossible work before copying identity/tokens into a flight
        // or consulting the cache. A warm hit must not bypass these contracts.
        exl3_require_prefill_width(prefill_rows);
        if(exact.max_context()<1 || prompt.size()>static_cast<std::size_t>(exact.max_context()))
            throw std::invalid_argument("serving prompt exceeds context capacity");
        if(prompt.empty() || cacheable_prefix_tokens>prompt.size() ||
            (cacheable_prefix_tokens && cacheable_prefix_tokens<policy_.minimum_prefix_tokens))
            throw std::invalid_argument("serving cacheable prefix extent");
        std::string contract;
        std::uint64_t epoch=0;
        {
            std::unique_lock service_lock(service_mutex_);
            contract=contract_;epoch=service_epoch_;
        }
        const auto require_epoch=[&] {
            if(epoch!=service_epoch_ || contract_!=contract)
                throw std::runtime_error("serving prefix preparation epoch changed");
        };
        const auto extend=[&](const auto& root,std::span<const std::int64_t> suffix) {
            if(!root->compact_draft())return root->append_prompt(exact,suffix,prefill_rows,progress);
            if(!draft || !staging)throw std::invalid_argument("compact cached prefix needs resident drafter");
            return root->append_prompt_compact(exact,*draft,*staging,suffix,prefill_rows,progress);
        };
        Result output;auto& metrics=output.metrics;metrics.prompt_tokens=prompt.size();
        metrics.configured_budget_bytes=policy_.byte_budget;
        std::shared_ptr<const Exl3VeriCacheRequest> prefix;
        const auto lookup_start=std::chrono::steady_clock::now();
        if(cacheable_prefix_tokens) prefix=index_.lookup_longest(exact,
            reuse_complete_input?prompt:prompt.first(cacheable_prefix_tokens),contract,policy_.minimum_prefix_tokens);
        if(prefix && reuse_complete_input)
            cacheable_prefix_tokens=std::max(cacheable_prefix_tokens,static_cast<std::size_t>(prefix->state()->position()));
        metrics.lookup_ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-lookup_start).count();
        metrics.cache_hit=static_cast<bool>(prefix);
        metrics.reused_prompt_tokens=prefix?static_cast<std::size_t>(prefix->state()->position()):0;
        metrics.executed_prompt_tokens=prompt.size()-metrics.reused_prompt_tokens;
        const auto prefill_start=std::chrono::steady_clock::now();
        if(cacheable_prefix_tokens) {
            bool flight_admission_resolved=false;
            if(!prefix || static_cast<std::size_t>(prefix->state()->position())<cacheable_prefix_tokens) {
                const auto source=prefix;
                auto prepared=prepare_shared_text_prefix(exact,prompt.first(cacheable_prefix_tokens),
                    contract,epoch,prefill_rows,progress,[&](const auto& root) {
                        // The helper holds service_mutex_ through admission and
                        // flight publication, so consumers cannot observe a
                        // completed flight before its admission is resolved.
                        const auto elapsed=std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now()-prefill_start).count();
                        const auto admission=index_.admit(root,contract,policy_.byte_budget,
                            available_physical_bytes,policy_.physical_reserve_bytes);
                        // A cached-root extension measures only its suffix, not
                        // a complete cold preparation cost.
                        if(admission.admitted && !source && elapsed>=0)
                            index_.supply_retention_observation(root,admission.generation,
                                static_cast<std::uint64_t>(elapsed),std::nullopt);
                        metrics.admitted=admission.admitted;metrics.replaced=admission.replaced;
                        metrics.evicted=admission.evicted;
                        metrics.effective_budget_bytes=admission.effective_budget_bytes;
                        metrics.cache_storage=admission.storage;
                    },[&](const auto& shared_progress) {
                        if(!source)return Exl3VeriCacheRequest::initialize(exact,
                            prompt.first(cacheable_prefix_tokens),prefill_rows,shared_progress);
                        const auto suffix=prompt.subspan(source->state()->position(),
                            cacheable_prefix_tokens-source->state()->position());
                        if(!source->compact_draft())
                            return source->append_prompt(exact,suffix,prefill_rows,shared_progress);
                        if(!draft || !staging)
                            throw std::invalid_argument("compact cached prefix needs resident drafter");
                        return source->append_prompt_compact(exact,*draft,*staging,
                            suffix,prefill_rows,shared_progress);
                    });
                flight_admission_resolved=!prepared.cache_hit;
                prefix=std::move(prepared.root);
                metrics.preparation_reused=prepared.joined;
                metrics.cache_hit=metrics.cache_hit || prepared.cache_hit;
                if(prepared.joined || prepared.cache_hit) {
                    exact.restore_exact_host_state(*prefix->state());
                    metrics.reused_prompt_tokens=cacheable_prefix_tokens;
                    metrics.executed_prompt_tokens=prompt.size()-cacheable_prefix_tokens;
                }
            }
            else exact.restore_exact_host_state(*prefix->state());
            if(!flight_admission_resolved && metrics.cache_hit &&
                static_cast<std::size_t>(prefix->state()->position())==cacheable_prefix_tokens &&
                metrics.reused_prompt_tokens<cacheable_prefix_tokens) {
                const auto cold_preparation_us=std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now()-prefill_start).count();
                std::unique_lock service_lock(service_mutex_);require_epoch();
                const auto admission=index_.admit(prefix,contract,policy_.byte_budget,
                    available_physical_bytes,policy_.physical_reserve_bytes);
                if(admission.admitted && !metrics.cache_hit && !metrics.preparation_reused && cold_preparation_us>=0)
                    index_.supply_retention_observation(prefix,admission.generation,
                        static_cast<std::uint64_t>(cold_preparation_us),std::nullopt);
                metrics.admitted=admission.admitted;metrics.replaced=admission.replaced;
                metrics.evicted=admission.evicted;metrics.effective_budget_bytes=admission.effective_budget_bytes;
                metrics.cache_storage=admission.storage;
            }
        }
        const auto attach_start=std::chrono::steady_clock::now();
        if(!cacheable_prefix_tokens) output.request=Exl3VeriCacheRequest::initialize(
            exact,prompt,prefill_rows,progress);
        else if(cacheable_prefix_tokens<prompt.size()) output.request=extend(
            prefix,prompt.subspan(cacheable_prefix_tokens));
        else output.request=prefix;
        metrics.attach_ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-attach_start).count();
        metrics.prefill_ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-prefill_start).count();
        {
            std::unique_lock service_lock(service_mutex_);require_epoch();
            // Bind fallback accounting to the same namespace epoch as the
            // returned root. Reload must not substitute a new cache's metadata
            // after the final identity check.
            if(!metrics.cache_storage.entries)metrics.cache_storage=index_.storage_stats();
        }
        if(!metrics.effective_budget_bytes) metrics.effective_budget_bytes=
            available_physical_bytes>policy_.physical_reserve_bytes?
                std::min(policy_.byte_budget,available_physical_bytes-policy_.physical_reserve_bytes):0;
        return output;
    }

    // Admit an immutable, already-authoritative closed-turn root. The caller
    // owns the semantic decision that the turn is complete and cacheable; this
    // layer only enforces ordinary-FP16 representation, token/state extent,
    // compatibility identity, cache bytes and the supplied physical reserve.
    // A concurrent model reload is serialized by the same service mutex.
    Exl3VeriCachePrefixIndex::Admission admit_completed_authority(
        std::shared_ptr<const Exl3VeriCacheRequest> root,
        std::uint64_t available_physical_bytes) {
        if(!root || !root->state() ||
            root->token_count()<policy_.minimum_prefix_tokens)
            throw std::invalid_argument("completed serving prefix extent");
        std::unique_lock service_lock(service_mutex_);
        return index_.admit(std::move(root),contract_,policy_.byte_budget,
            available_physical_bytes,policy_.physical_reserve_bytes);
    }
    // The caller supplies the entire explicitly rendered input, distinct from
    // declaring an arbitrary generated suffix to be a completed conversation.
    Exl3VeriCachePrefixIndex::Admission admit_input_authority(
        std::shared_ptr<const Exl3VeriCacheRequest> root,std::span<const std::int64_t> input,
        std::uint64_t available_physical_bytes,
        std::optional<std::span<const Exl3VeriCachePrefixIndex::RetentionDecision>> retention=std::nullopt,
        std::optional<std::uint64_t> preparation_microseconds=std::nullopt) {
        if(!root || root->prepared_identity() || input.size()<policy_.minimum_prefix_tokens ||
            !root->matches_tokens(input))
            throw std::invalid_argument("input authority exact token identity");
        std::unique_lock lock(service_mutex_);
        const auto observed_root=root;
        auto admission=index_.admit(std::move(root),contract_,policy_.byte_budget,
            available_physical_bytes,policy_.physical_reserve_bytes,retention);
        if(admission.admitted && preparation_microseconds)
            index_.supply_retention_observation(observed_root,admission.generation,
                preparation_microseconds,std::nullopt);
        return admission;
    }

    // One-shot cold concurrent preparation fault after numerical completion,
    // before index mutation. Warm hits do not consume it.
    void fail_next_preparation_admission_for_test() {
        std::lock_guard lock(service_mutex_);
        if(fail_preparation_admission_)throw std::logic_error("preparation admission fault already armed");
        fail_preparation_admission_=true;
    }
    void reset_for_model_reload(Exl3VeriCacheServingIdentity identity) {
        auto next=identity.contract();std::unique_lock lock(service_mutex_);
        if(service_epoch_==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("serving prefix reload epoch overflow");
        index_.clear();contract_=std::move(next);++service_epoch_;
        // Wake old-namespace consumers without releasing in-flight ownership.
        // Keep occupied slots until their producers retire: repeated reloads
        // must not turn the bounded registry into unbounded outstanding fills.
        for(const auto& flight:preparations_)if(flight)flight->changed.notify_all();
    }
    Exl3VeriCachePrefixIndex::StorageStats trim(std::uint64_t available_physical_bytes,
        std::size_t* evicted=nullptr) {
        std::unique_lock lock(service_mutex_);
        return index_.trim_to_budget(policy_.byte_budget,available_physical_bytes,
            policy_.physical_reserve_bytes,evicted);
    }
    // One-shot optional policy input; never retain a stale decision for a later
    // admission. Admission/reload cannot interleave with this cache transaction.
    // Required publication coverage remains owned outside the retention index.
    Exl3VeriCachePrefixIndex::PolicyTrim trim_with_retention_policy(
        std::uint64_t available_physical_bytes,
        std::span<const Exl3VeriCachePrefixIndex::RetentionDecision> decisions) {
        std::unique_lock lock(service_mutex_);
        return index_.trim_with_retention_policy(policy_.byte_budget,available_physical_bytes,
            policy_.physical_reserve_bytes,decisions);
    }
    Exl3VeriCachePrefixIndex::StorageStats storage_stats() const {return index_.storage_stats();}
    std::vector<Exl3VeriCachePrefixIndex::RetentionMetadata> retention_metadata() const {
        return index_.retention_metadata();
    }
    std::vector<std::shared_ptr<const Exl3VeriCacheRequest>> roots() const {return index_.roots();}
    std::size_t root_capacity() const noexcept {return index_.capacity();}
    void copy_roots_into(std::vector<std::shared_ptr<const Exl3VeriCacheRequest>>& output) const {
        index_.copy_roots_into(output);
    }
private:
    Policy policy_;
    struct PrefixPreparation {
        std::shared_ptr<const void> model;
        // Service-local monotonic epoch binds the contract; reload changes it
        // under the same mutex and overflow refuses before namespace mutation.
        std::uint64_t epoch=0;
        std::vector<std::int64_t> tokens;
        int prefill_rows=0;
        std::thread::id producer;
        std::atomic<std::size_t> waiters{0};
        std::size_t participants=0; // service_mutex_, independent of inventory owners
        bool done=false;
        std::shared_ptr<const Exl3VeriCacheRequest> root;
        std::exception_ptr failure;
        std::condition_variable changed;
    };
public:
    // Startup sizing for the real flight objects and their owned token payload.
    // Inline registry storage already belongs to the cache object. Allocator
    // bookkeeping is not represented as payload, matching inventory conventions.
    static Exl3ResourceInventory::Requirement preparation_storage_requirement(
        std::size_t slots,std::size_t maximum_tokens) {
        if(!slots || slots>8 || !maximum_tokens || maximum_tokens>static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("preparation storage extent");
        Exl3ResourceInventory::Requirement required;required.configuration=0x50524550464C;
        required.add(Exl3ResourceInventory::Domain::host_metadata,slots,sizeof(PrefixPreparation));
        required.add(Exl3ResourceInventory::Domain::host_metadata,slots,
            static_cast<std::uint64_t>(maximum_tokens)*sizeof(std::int64_t));
        return required;
    }
    // Caller serializes startup before exposing cache to any request/reload.
    // Publish the pool only after the coordinator commits its real ownership.
    template<class Authority> void reserve_preparation_storage(Authority& authority,
        std::size_t slots,std::size_t maximum_tokens,unsigned fail_after_slot_for_test=0,
        std::array<std::weak_ptr<const void>,8>* constructed_owners_for_test=nullptr) {
        const auto required=preparation_storage_requirement(slots,maximum_tokens);
        // slots+1 reports a one-byte inventory mismatch after full construction.
        if(fail_after_slot_for_test>slots+1)throw std::invalid_argument("preparation storage fault extent");
        if(preparation_pool_enabled_)throw std::logic_error("preparation storage already reserved");
        for(const auto& flight:preparations_)if(flight)
            throw std::logic_error("preparation storage requires idle startup");
        std::array<std::shared_ptr<PrefixPreparation>,8> staged{};
        authority.allocate_startup_resources(required,[&](std::uint64_t configuration) {
            if(configuration!=required.configuration)throw std::logic_error("preparation storage credit identity");
            Exl3ResourceInventory actual;
            for(std::size_t i=0;i<slots;++i) {
                auto flight=std::make_shared<PrefixPreparation>();
                flight->tokens.reserve(maximum_tokens);
                actual.add({flight,0,Exl3ResourceInventory::Domain::host_metadata,
                    sizeof(PrefixPreparation)+flight->tokens.capacity()*sizeof(std::int64_t)-
                        ((fail_after_slot_for_test==slots+1 && i==0)?1:0)});
                staged[i]=std::move(flight);
                if(constructed_owners_for_test)(*constructed_owners_for_test)[i]=staged[i];
                if(fail_after_slot_for_test==i+1)
                    throw std::runtime_error("injected preparation storage construction failure");
            }
            return actual;
        },[&]() noexcept {staged={};});
        preparation_pool_=std::move(staged);preparation_pool_enabled_=true;
    }
    struct PreparationStorageStats {
        std::size_t slots=0,participants=0,retained_roots=0;
        std::uint64_t metadata_bytes=0;
    };
    PreparationStorageStats preparation_storage_stats() const {
        std::lock_guard lock(service_mutex_);PreparationStorageStats result;
        for(const auto& flight:preparation_pool_)if(flight) {
            ++result.slots;result.participants+=flight->participants;
            if(flight->root)++result.retained_roots;
            result.metadata_bytes+=sizeof(PrefixPreparation)+flight->tokens.capacity()*sizeof(std::int64_t);
        }
        return result;
    }
private:
    std::array<std::shared_ptr<PrefixPreparation>,8> preparations_{};
    std::array<std::shared_ptr<PrefixPreparation>,8> preparation_pool_{};
    bool preparation_pool_enabled_=false;
    struct PreparedPrefix {
        std::shared_ptr<const Exl3VeriCacheRequest> root;
        bool joined=false;
        bool cache_hit=false;
    };
    template<class Build> PreparedPrefix prepare_shared_text_prefix(
        Exl3TextContext& exact,std::span<const std::int64_t> tokens,
        const std::string& contract,std::uint64_t epoch,int prefill_rows,
        const std::function<void(int)>& progress,
        const std::function<void(const std::shared_ptr<const Exl3VeriCacheRequest>&)>& admit,
        Build&& build) {
        std::shared_ptr<PrefixPreparation> flight;
        bool producer=false;
        struct Participant {
            Exl3VeriCacheServingPrefixCache& cache;
            std::shared_ptr<PrefixPreparation> flight;
            ~Participant() {
                if(!flight)return;
                std::lock_guard lock(cache.service_mutex_);
                if(--flight->participants==0) {
                    flight->root.reset();flight->model.reset();flight->failure={};
                    flight->tokens.clear();flight->done=false;
                }
            }
        } participant{*this,{}};
        {
            std::unique_lock lock(service_mutex_);
            if(epoch!=service_epoch_ || contract!=contract_)
                throw std::runtime_error("serving prefix preparation epoch changed");
            // Initial lookup ran outside this lock. A producer may have admitted
            // and removed its flight since then. Recheck exact authority before
            // reserving a new producer, atomically with flight registration.
            if(auto cached=index_.lookup(exact,tokens,contract))
                return {std::move(cached),false,true};
            for(const auto& current:preparations_)
                if(current && current->epoch==epoch &&
                    current->model==exact.model_identity() && current->prefill_rows==prefill_rows &&
                    current->tokens.size()==tokens.size() &&
                    std::equal(tokens.begin(),tokens.end(),current->tokens.begin())) {flight=current;break;}
            if(!flight)for(auto& slot:preparations_)if(!slot) {
                std::shared_ptr<PrefixPreparation> created;
                if(preparation_pool_enabled_) {
                    for(const auto& available:preparation_pool_)
                        if(available && !available->participants && available->tokens.capacity()>=tokens.size()) {
                            created=available;break;
                        }
                    if(!created)break; // independent work, never uncredited flight growth
                } else created=std::make_shared<PrefixPreparation>();
                created->model=exact.model_identity();created->epoch=epoch;
                created->tokens.assign(tokens.begin(),tokens.end());created->prefill_rows=prefill_rows;
                created->producer=std::this_thread::get_id();slot=created;flight=std::move(created);
                ++flight->participants;participant.flight=flight;producer=true;break;
            }
            if(flight && !producer) {
                if(flight->producer==std::this_thread::get_id())
                    throw std::logic_error("recursive identical prefix preparation");
                ++flight->participants;participant.flight=flight;
                struct WaiterOwner {
                    PrefixPreparation& flight;
                    explicit WaiterOwner(PrefixPreparation& value):flight(value) {
                        flight.waiters.fetch_add(1,std::memory_order_relaxed);
                    }
                    ~WaiterOwner(){flight.waiters.fetch_sub(1,std::memory_order_relaxed);}
                } waiter(*flight);
                while(!flight->done && epoch==service_epoch_ && contract==contract_) {
                    lock.unlock();if(progress)progress(0);lock.lock();
                    if(epoch!=service_epoch_ || contract!=contract_)
                        throw std::runtime_error("serving prefix preparation epoch changed");
                    flight->changed.wait_for(lock,std::chrono::milliseconds(10),[&]{
                        return flight->done || epoch!=service_epoch_ || contract!=contract_;
                    });
                }
                // Completion and reload may both happen before this waiter
                // reacquires the lock. A completed old root is still stale.
                if(epoch!=service_epoch_ || contract!=contract_)
                    throw std::runtime_error("serving prefix preparation epoch changed");
                if(flight->failure)std::rethrow_exception(flight->failure);
                // Completion may wake us before the next timed cancellation
                // poll. Let this consumer decline the result without revoking
                // the successful shared root or another consumer's ownership.
                lock.unlock();if(progress)progress(0);lock.lock();
                if(epoch!=service_epoch_ || contract!=contract_)
                    throw std::runtime_error("serving prefix preparation epoch changed");
                return {flight->root,true};
            }
        }
        // Bounded saturation falls back to independent caller-owned work.
        if(!flight) {
            auto root=build(progress);
            std::unique_lock lock(service_mutex_);
            if(epoch!=service_epoch_ || contract!=contract_)
                throw std::runtime_error("serving prefix preparation epoch changed");
            if(std::exchange(fail_preparation_admission_,false))throw std::bad_alloc();
            admit(root);return {std::move(root),false};
        }
        std::exception_ptr producer_callback_failure;
        try {
            const auto shared_progress=[&](int position) {
                if(!producer_callback_failure && progress) {
                    try {progress(position);}
                    catch(...) {producer_callback_failure=std::current_exception();}
                }
                // A callback exception belongs to this consumer. Keep its
                // context alive synchronously while another consumer needs the
                // fill; numerical failures still fail the whole preparation.
                if(producer_callback_failure && !flight->waiters.load(std::memory_order_relaxed))
                    std::rethrow_exception(producer_callback_failure);
            };
            auto root=build(shared_progress);
            std::unique_lock lock(service_mutex_);
            if(epoch!=service_epoch_ || contract!=contract_)
                throw std::runtime_error("serving prefix preparation epoch changed");
            if(std::exchange(fail_preparation_admission_,false))throw std::bad_alloc();
            admit(root);
            flight->root=root;flight->done=true;
            for(auto& slot:preparations_)if(slot==flight){slot.reset();break;}
            flight->changed.notify_all();
            lock.unlock();
            if(producer_callback_failure)std::rethrow_exception(producer_callback_failure);
            return {std::move(root),false};
        } catch(...) {
            std::unique_lock lock(service_mutex_);
            // A producer's deferred callback error must not overwrite an
            // already published successful result for surviving consumers.
            if(flight->done)throw;
            flight->failure=std::current_exception();flight->done=true;
            for(auto& slot:preparations_)if(slot==flight){slot.reset();break;}
            flight->changed.notify_all();throw;
        }
    }
    std::string contract_;
    Exl3VeriCachePrefixIndex index_;
    mutable std::mutex service_mutex_;
    std::uint64_t service_epoch_=1;
    bool fail_preparation_admission_=false;
};

// Cold request representation: token pages plus either reset identity or one
// explicitly bound shared-prefix checkpoint. No chain of full recurrent images.
// Dropping the caller's hot request releases its private KV/GDN/tap state.
class Exl3VeriCacheReplay {
    Exl3TokenHistory history_;
    std::shared_ptr<const void> model_identity_;
    std::shared_ptr<const Exl3VeriCacheRequest> anchor_;
    std::vector<std::pair<int,int>> partitions_;
    friend class Exl3VeriCacheRequest;
    Exl3VeriCacheReplay()=default;
public:
    int position() const noexcept {return history_.position();}
    int checkpoint_position() const noexcept {return anchor_?anchor_->state()->position():0;}
    std::uint64_t token_allocated_bytes() const {
        const std::array<const Exl3TokenHistory*,1> histories{&history_};
        return Exl3TokenHistory::visit(histories).allocated_bytes;
    }
    std::shared_ptr<const Exl3VeriCacheRequest> restore(Exl3TextContext& exact,int prefill_rows=1024,
        const std::function<void(int)>& progress={}) const {
        Exl3VeriCacheRequest::validate_prefill_width(prefill_rows);
        if(exact.model_identity()!=model_identity_ || position()>exact.max_context())
            throw std::invalid_argument("cold replay numerical reference or position mismatch");
        try {
        const auto tokens=history_.suffix(0);
        const auto view=std::span<const std::int64_t>(tokens);
        auto current=anchor_;
        int cursor=checkpoint_position();
        if(partitions_.empty()) throw std::logic_error("cold replay draft partitions missing");
        // Prefix target computation can use qualified wide rows. The bounded
        // retained draft window replays its actual original call partitions;
        // regrouping these calls could alter draft projection rounding.
        const int first=partitions_.front().first;
        if(cursor<first) {
            current=current?current->append_prompt(exact,view.subspan(cursor,first-cursor),prefill_rows,progress):
                Exl3VeriCacheRequest::initialize(exact,view.first(first),prefill_rows,progress);
            cursor=first;
        } else if(current) exact.restore_exact_host_state(*current->state_);
        auto restored=current?Exl3VeriCacheRequest::create_planned(current.get(),
            static_cast<std::size_t>(position()-cursor),true,exact.request_metadata_reservation(),exact.request_metadata_reservation()):nullptr;
        for(const auto& [begin,rows]:partitions_) {
            if(begin+rows<=cursor) continue;
            if(begin!=cursor) throw std::logic_error("cold replay checkpoint cuts a draft partition");
            if(restored) restored->ingest_prompt_suffix(exact,view.subspan(begin,rows),8,progress);
            else {
                current=Exl3VeriCacheRequest::initialize(exact,view.subspan(begin,rows),8,progress);
                restored=Exl3VeriCacheRequest::create_planned(current.get(),
                    static_cast<std::size_t>(position()-begin-rows),true,exact.request_metadata_reservation(),exact.request_metadata_reservation());
            }
            cursor+=rows;
        }
        if(!restored || cursor!=position() || restored->blocks_.size()!=partitions_.size())
            throw std::logic_error("cold replay final state/partition extent");
        for(std::size_t i=0;i<partitions_.size();++i)
            if(restored->blocks_[i]->first!=partitions_[i].first || restored->blocks_[i]->rows!=partitions_[i].second)
                throw std::logic_error("cold replay changed retained draft partitions");
        restored->state_=exact.export_exact_host_state();
        // create_planned already supplied a fresh revision; retain it rather
        // than allocating and immediately discarding a second identity.
        restored->parent_revision_=anchor_?anchor_->revision_:nullptr;
        restored->replay_anchor_revision_=restored->parent_revision_;
        restored->history_=history_;
        return restored;
        } catch(...) {
            const auto failure=std::current_exception();
            try{exact.reset();}catch(...){} // The immutable cold plan remains the retry authority.
            std::rethrow_exception(failure);
        }
    }
};

// Owning typed replay authority. Unlike Exl3VeriCacheReplay this retains the
// exact encoder inputs and frontend-resolved media partitions/MRoPE positions.
// It is inert until the existing prepared numerical caller consumes it.
class Exl3PreparedMediaReplay {
    targets::qwen3_6::PreparedPromptData data_;
    targets::qwen3_6::VisionControl control_;
    std::shared_ptr<const Exl3VeriCacheRequest> ancestor_;
    Exl3PreparedIdentity identity_;
    explicit Exl3PreparedMediaReplay(targets::qwen3_6::PreparedPromptData data,
        targets::qwen3_6::VisionControl control,
        std::shared_ptr<const Exl3VeriCacheRequest> ancestor)
        :data_(std::move(data)),control_(std::move(control)),ancestor_(std::move(ancestor)),
         identity_(data_) {}
public:
    static Exl3PreparedMediaReplay create(
        const targets::qwen3_6::PreparedPromptData& source,
        std::shared_ptr<const Exl3VeriCacheRequest> ancestor={}) {
        if(!source.has_media() || !source.media_payload_identity_valid() ||
           source.token_ids.empty() || source.token_types.size()!=source.token_ids.size() ||
           source.positions.size()!=source.token_ids.size()*3)
            throw std::invalid_argument("typed media replay source extent");
        auto data=source;
        const auto plan=targets::qwen3_6::plan_vision_control(data);
        auto control=targets::qwen3_6::build_vision_control(data,plan,0);
        if(control.items.size()!=data.vision_items.size())
            throw std::invalid_argument("typed media replay partition extent");
        Exl3PreparedIdentity identity(data);
        if(ancestor) {
            if(!ancestor->prepared_identity() || !ancestor->state() ||
               ancestor->token_count()>=data.token_ids.size() ||
               ancestor->state()->position()!=static_cast<int>(ancestor->token_count()) ||
               ancestor->state()->rope_offset()!=ancestor->prepared_identity()->rope_delta() ||
               !ancestor->prepared_identity()->prefix_positions_equal(identity,ancestor->token_count()))
                throw std::invalid_argument("typed media replay ancestor proof");
            const auto frontier=ancestor->token_count();
            if(data.token_types[frontier]==0)for(std::size_t axis=0;axis<3;++axis)
                if(data.positions[axis*data.token_ids.size()+frontier]!=
                   static_cast<std::int64_t>(frontier)+ancestor->state()->rope_offset())
                    throw std::invalid_argument("typed media replay suffix position proof");
        }
        return Exl3PreparedMediaReplay(std::move(data),std::move(control),std::move(ancestor));
    }
    [[nodiscard]] const targets::qwen3_6::PreparedPromptData& data() const noexcept{return data_;}
    [[nodiscard]] const targets::qwen3_6::VisionControl& control() const noexcept{return control_;}
    [[nodiscard]] const std::shared_ptr<const Exl3VeriCacheRequest>& ancestor() const noexcept{return ancestor_;}
    [[nodiscard]] const Exl3PreparedIdentity& identity() const noexcept{return identity_;}
    [[nodiscard]] bool matches_source(const targets::qwen3_6::PreparedPromptData& source) const {
        if(!source.media_payload_identity_valid() || source.media_payloads.size()!=data_.media_payloads.size())return false;
        Exl3PreparedIdentity candidate(source);
        if(!identity_.equals(candidate))return false;
        for(std::size_t i=0;i<source.media_payloads.size();++i)
            if(source.media_payloads[i].get()!=data_.media_payloads[i].get())return false;
        const auto plan=targets::qwen3_6::plan_vision_control(source);
        const auto control=targets::qwen3_6::build_vision_control(source,plan,0);
        if(control.items.size()!=control_.items.size())return false;
        for(std::size_t i=0;i<control.items.size();++i)
            if(!exl3_same_vision_control(control.items[i],control_.items[i]))return false;
        return true;
    }
};

inline Exl3VeriCacheReplay Exl3VeriCacheRequest::replay_plan(
    std::shared_ptr<const Exl3VeriCacheRequest> anchor) const {
    if(prepared_identity_ || (anchor && anchor->prepared_identity_))
        throw std::logic_error("prepared media cannot be reconstructed by token-only replay");
    if(history_.position()!=state_->position()) throw std::logic_error("authoritative token/state position mismatch");
    if(anchor && (anchor->revision_!=replay_anchor_revision_ || anchor->state_->position()>=state_->position()))
        throw std::invalid_argument("replay checkpoint is not the bound prefix ancestor");
    if(anchor && anchor->projected_) throw std::invalid_argument("compact prefix-anchor replay needs explicit draft streaming");
    Exl3VeriCacheReplay result;result.history_=history_;result.model_identity_=state_->model_identity();
    result.partitions_=partitions_;
    result.anchor_=std::move(anchor);return result;
}
} // namespace ninfer::exl3
