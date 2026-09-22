#pragma once
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"
#include <memory>
#include <algorithm>
#include <stdexcept>
#include "exl3/bounded_shared_owner.h"
#include "exl3/resource_inventory.h"

namespace ninfer::exl3 {
// Frontend-resolved boundaries only; never infer a tool boundary from tokens or
// trim/normalize tool text. Zero means no eligible shared text checkpoint.
inline std::size_t exl3_declared_prefix_frontier(
    const targets::qwen3_6::PreparedPromptData& prompt,std::size_t upper) {
    if(!prompt.identity.reusable || prompt.has_media())return 0;
    upper=std::min(upper,prompt.token_ids.size());
    if(prompt.identity.rewrite_checkpoint)
        upper=std::min<std::size_t>(upper,prompt.identity.rewrite_checkpoint->frontier);
    std::size_t selected=0;
    for(const auto& opportunity:prompt.context_cache.opportunities) {
        if(opportunity.frontier>prompt.token_ids.size())
            throw std::invalid_argument("declared prefix frontier exceeds prepared input");
        if(opportunity.kind==PromptCacheMarkerKind::SharedStablePrefix &&
            opportunity.frontier>=64 && opportunity.frontier<=upper)
            selected=std::max<std::size_t>(selected,opportunity.frontier);
    }
    return selected;
}
// Exact typed metadata, separate from the token history and numerical state.
// Only the owning prepared-input execution path may install this in a root.
// Identity-only retention: ResidentPrefixIdentity copies VisionItem descriptors,
// digests and positions, never PreparedPromptData::media_payloads. This object
// cannot authorize media replay by itself; replay must retain its payload owner.
class Exl3PreparedIdentity {
    struct AppendKey {};
    targets::qwen3_6::detail::ResidentPrefixIdentity identity_;
    std::int32_t rope_delta_ = 0;
    bool reusable_ = false;
public:
    Exl3PreparedIdentity(AppendKey,const Exl3PreparedIdentity& source,std::size_t count)
        :identity_(source.identity_.appended_copy(count,source.rope_delta_)),
         rope_delta_(source.rope_delta_),reusable_(source.reusable_) {}
    static std::shared_ptr<Exl3PreparedIdentity> create(const targets::qwen3_6::PreparedPromptData& prompt,
        const std::function<RetainedDescriptorLedger::Ticket(std::uint64_t)>& reserve={}) {
        Exl3ResourceInventory::Requirement required;using Domain=Exl3ResourceInventory::Domain;
        required.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<Exl3PreparedIdentity>());
        const auto add=[&](std::size_t count,std::size_t width){if(count)required.add(Domain::host_metadata,count,width);};
        add(prompt.token_types.size(),sizeof(prompt.token_types[0]));
        add(prompt.positions.size(),sizeof(std::int32_t));
        add(prompt.vision_items.size(),sizeof(prompt.vision_items[0]));
        for(const auto& item:prompt.vision_items) {
            add(item.timestamps.size(),sizeof(item.timestamps[0]));
            add(item.token_spans.size(),sizeof(item.token_spans[0]));
        }
        add(prompt.identity.rewrite_execution_frontiers.size(),sizeof(prompt.identity.rewrite_execution_frontiers[0]));
        std::optional<RetainedDescriptorLedger::Ticket> credit;
        if(reserve) {
            const auto bytes=required.units[static_cast<unsigned>(Domain::host_metadata)];
            credit.emplace(reserve(bytes));
            if(credit->bytes()!=bytes)throw std::invalid_argument("prepared identity reservation extent");
        }
        auto result=make_bounded_shared<Exl3PreparedIdentity>(prompt);
        if(credit && credit->bytes()!=result->metadata_bytes())throw std::logic_error("prepared identity allocation extent");
        if(credit && !attach_metadata_credit(result,std::move(*credit)))throw std::logic_error("prepared identity credit attachment");
        return result;
    }
    std::uint64_t metadata_bytes() const {
        Exl3ResourceInventory::Requirement required;using Domain=Exl3ResourceInventory::Domain;
        required.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<Exl3PreparedIdentity>());
        identity_.visit_allocations([&](const void*,std::size_t bytes){
            if(bytes)required.add(Domain::host_metadata,1,bytes);
        },true);
        return required.units[static_cast<unsigned>(Domain::host_metadata)];
    }
    static bool metadata_credit_belongs_to(const std::shared_ptr<const void>& owner,
        const RetainedDescriptorLedger& ledger) noexcept {
        return bounded_split_credit_belongs_to<Exl3PreparedIdentity>(owner,ledger);
    }
    static bool attach_metadata_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        if(!owner || !owner.use_count())return false;
        try {
            const auto* value=static_cast<const Exl3PreparedIdentity*>(owner.get());
            return attach_bounded_split_retirement_credit<Exl3PreparedIdentity>(owner,std::move(credit),
                value->metadata_bytes()-bounded_shared_allocation_bytes<Exl3PreparedIdentity>());
        }catch(...){return false;}
    }
    explicit Exl3PreparedIdentity(const targets::qwen3_6::PreparedPromptData& prompt)
        : rope_delta_(prompt.rope_delta), reusable_(prompt.identity.reusable) {
        identity_.assign(prompt);
        if (!identity_.matches(prompt, prompt.token_ids.size()))
            throw std::invalid_argument("prepared identity splits a media item");
    }
    std::size_t size() const noexcept { return identity_.size(); }
    bool reusable() const noexcept { return reusable_; }
    std::int32_t rope_delta() const noexcept { return rope_delta_; }
    bool prefix_equals(const Exl3PreparedIdentity& other, std::size_t count) const {
        // Retain the conservative whole-offset check until typed ancestor
        // restoration with a changed final MRoPE offset is qualified.
        return reusable_ && other.reusable_ && rope_delta_ == other.rope_delta_ &&
            identity_.prefix_equals(other.identity_, count);
    }
    // Typed ancestor proof compares every retained token type, media item,
    // rewrite frontier and position axis. The final prompt rope delta may grow
    // after the ancestor; the caller must separately bind the ancestor state's
    // authoritative rope offset before numerical restoration.
    bool prefix_positions_equal(const Exl3PreparedIdentity& other,std::size_t count) const {
        return reusable_ && other.reusable_ && identity_.prefix_equals(other.identity_,count);
    }
    bool equals(const Exl3PreparedIdentity& other) const {
        return size() == other.size() && prefix_equals(other, size());
    }
    std::shared_ptr<const Exl3PreparedIdentity> append_text(std::size_t count,
        const std::function<RetainedDescriptorLedger::Ticket(std::uint64_t)>& reserve={}) const {
        std::optional<RetainedDescriptorLedger::Ticket> credit;
        if(reserve) {
            Exl3ResourceInventory::Requirement required;using Domain=Exl3ResourceInventory::Domain;
            required.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<Exl3PreparedIdentity>());
            const auto dynamic=identity_.copied_append_allocation_bytes(count,rope_delta_);
            if(dynamic)required.add(Domain::host_metadata,1,dynamic);
            const auto bytes=required.units[static_cast<unsigned>(Domain::host_metadata)];
            credit.emplace(reserve(bytes));
            if(credit->bytes()!=bytes)throw std::invalid_argument("prepared identity append reservation extent");
        }
        auto next = make_bounded_shared<Exl3PreparedIdentity>(AppendKey{},*this,count);
        if(credit && credit->bytes()!=next->metadata_bytes())throw std::logic_error("prepared identity append allocation extent");
        if(credit && !attach_metadata_credit(next,std::move(*credit)))throw std::logic_error("prepared identity append attachment");
        return next;
    }
    void visit_allocations(const std::function<void(const void*, std::size_t)>& visitor,
                           bool capacity = true) const {
        visitor(this, sizeof(*this));
        identity_.visit_allocations(visitor, capacity);
    }
};
} // namespace ninfer::exl3
