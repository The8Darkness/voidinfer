#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <variant>
#include <vector>
#include "exl3/bounded_shared_owner.h"

namespace ninfer::exl3 {
// Immutable authoritative token record pages. Equal strings do not deduplicate.
// Explicit forks share both token pages and the bounded-depth index. Appending
// copies only a partial tail and its radix path; destruction is bounded-depth too.
class Exl3TokenHistory {
    struct Page {int rows=0;std::array<std::int64_t,64> tokens{};};
    struct Node;
    using Edges=std::array<std::shared_ptr<const Node>,32>;
    struct Node {
        std::variant<Page,Edges> data;
        explicit Node(Page page):data(std::move(page)){}
        explicit Node(Edges edges):data(std::move(edges)){}
    };
    std::shared_ptr<const Node> root_;
    unsigned level_=0; // At most five radix edges for INT32_MAX tokens.
    inline static thread_local unsigned allocation_fault_countdown_=0;
    using MetadataReservation=std::function<RetainedDescriptorLedger::Ticket(std::uint64_t)>;
    template<class Value> static std::shared_ptr<const Node> make_node(Value value,const MetadataReservation& reserve) {
        if(allocation_fault_countdown_ && --allocation_fault_countdown_==0)throw std::bad_alloc();
        std::optional<RetainedDescriptorLedger::Ticket> credit;
        if(reserve) {
            credit.emplace(reserve(node_metadata_bytes()));
            if(credit->bytes()!=node_metadata_bytes())throw std::invalid_argument("token control metadata extent");
        }
        auto result=make_bounded_shared<Node>(std::move(value));
        if(credit && !attach_control_credit(result,std::move(*credit)))
            throw std::logic_error("token control metadata attachment");
        return result;
    }
    // Construct each affected index node once even when a cold prompt appends
    // many pages. Retained siblings remain untouched throughout construction.
    static std::shared_ptr<const Node> append_pages(const std::shared_ptr<const Node>& node,
        unsigned level,std::uint64_t base,std::uint64_t first,std::uint64_t end,
        std::span<const std::int64_t> tokens,const MetadataReservation& reserve,bool promoted_root=false) {
        const auto begin=std::max(base,first);
        const auto limit=std::min(base+(std::uint64_t{64}<<(5*level)),end);
        if(!level) {
            Page page=node?std::get<Page>(node->data):Page{};
            std::copy_n(tokens.begin()+static_cast<std::size_t>(begin-first),
                static_cast<std::size_t>(limit-begin),page.tokens.begin()+static_cast<std::size_t>(begin-base));
            page.rows=static_cast<int>(limit-base);
            return make_node(std::move(page),reserve);
        }
        Edges edges{};
        if(promoted_root)edges[0]=node;
        else if(node)edges=std::get<Edges>(node->data);
        const auto child_rows=std::uint64_t{64}<<(5*(level-1));
        for(auto slot=(begin-base)/child_rows;slot<=(limit-1-base)/child_rows;++slot)
            edges[slot]=append_pages(edges[slot],level-1,base+slot*child_rows,first,end,tokens,reserve);
        return make_node(std::move(edges),reserve);
    }
    const Node* page_node(std::uint32_t index) const noexcept {
        const Node* node=root_.get();
        for(unsigned level=level_;level;--level)
            node=std::get<Edges>(node->data)[(index>>(5*(level-1)))&31U].get();
        return node;
    }
    static bool same_nodes(const std::shared_ptr<const Node>& left,
        const std::shared_ptr<const Node>& right) noexcept {
        if(left==right)return true;
        if(!left || !right || left->data.index()!=right->data.index())return false;
        if(const auto* a=std::get_if<Page>(&left->data)) {
            const auto& b=std::get<Page>(right->data);
            return a->rows==b.rows && std::equal(a->tokens.begin(),a->tokens.begin()+a->rows,b.tokens.begin());
        }
        const auto& a=std::get<Edges>(left->data);const auto& b=std::get<Edges>(right->data);
        for(std::size_t i=0;i<a.size();++i)if(!same_nodes(a[i],b[i]))return false;
        return true;
    }
    int position_=0;
    std::uint64_t fingerprint_=1469598103934665603ULL;
public:
    static std::size_t live_node_blocks_for_test() noexcept {
        return bounded_shared_live_blocks_for_test<Node>();
    }
    static constexpr std::size_t node_metadata_bytes() noexcept {
        // The allocation retains its Node storage even after final strong
        // destruction. Token ranges separately budget OS-locked pages, not
        // this metadata allocation; keep the full block charged until weak zero.
        return bounded_shared_allocation_bytes<Node>();
    }
    static bool control_credit_belongs_to(const std::shared_ptr<const void>& owner,
        const RetainedDescriptorLedger& ledger) noexcept {
        return bounded_credit_belongs_to<Node>(owner,ledger);
    }
    static bool attach_control_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        if(credit.bytes()!=node_metadata_bytes())return false;
        return attach_bounded_retirement_credit<Node>(owner,std::move(credit));
    }
    template<class Visitor> void visit_control_owners(Visitor&& visitor) const {
        const auto walk=[&](auto&& self,const std::shared_ptr<const Node>& node)->void {
            if(!node)return;
            visitor(std::shared_ptr<const void>(node));
            if(const auto* edges=std::get_if<Edges>(&node->data))for(const auto& child:*edges)self(self,child);
        };
        walk(walk,root_);
    }
    // One-shot, thread-local construction fault. Invalid/empty appends consume
    // no allocation. Zero explicitly disarms a fixture before leaving its scope.
    static void fail_allocation_for_test(unsigned allocation) noexcept {
        allocation_fault_countdown_=allocation;
    }
    static std::uint64_t extend_fingerprint(std::uint64_t value,std::span<const std::int64_t> tokens) noexcept {
        for(auto token:tokens)for(int shift=0;shift<64;shift+=8) {
            value^=static_cast<std::uint8_t>(static_cast<std::uint64_t>(token)>>shift);
            value*=1099511628211ULL;
        }
        return value;
    }
    std::uint64_t fingerprint() const noexcept {return fingerprint_;}
    int position() const noexcept {return position_;}
    bool same_tokens(const Exl3TokenHistory& other) const noexcept {
        return position_==other.position_ && same_nodes(root_,other.root_);
    }
    bool equals(std::span<const std::int64_t> tokens) const noexcept {
        if(tokens.size()!=static_cast<std::size_t>(position_))return false;
        std::size_t offset=0;
        while(offset<tokens.size()) {
            const auto& page=std::get<Page>(page_node(static_cast<std::uint32_t>(offset/64))->data);
            if(!std::equal(page.tokens.begin(),page.tokens.begin()+page.rows,tokens.begin()+offset))return false;
            offset+=page.rows;
        }
        return offset==tokens.size();
    }
    Exl3TokenHistory append(std::span<const std::int64_t> tokens,const MetadataReservation& reserve={}) const {
        if(tokens.size()>static_cast<std::size_t>(INT32_MAX-position_)) throw std::invalid_argument("token history extent");
        for(auto token:tokens) if(token<0 || token>=248320) throw std::invalid_argument("token history token extent");
        auto result=*this;
        if(tokens.empty())return result;
        const auto end=static_cast<std::uint64_t>(position_)+tokens.size();
        bool promoted_root=false;
        while(end>(std::uint64_t{64}<<(5*result.level_))) {
            ++result.level_;
            // The final new root is constructed by append_pages. Materialize
            // only intermediate levels that the new root must retain as child 0.
            if(result.root_ && end>(std::uint64_t{64}<<(5*result.level_))) {
                Edges edges{};edges[0]=result.root_;
                result.root_=make_node(std::move(edges),reserve);
            } else promoted_root=bool(result.root_);
        }
        result.root_=append_pages(result.root_,result.level_,0,position_,end,tokens,reserve,promoted_root);
        result.position_=static_cast<int>(end);
        result.fingerprint_=extend_fingerprint(fingerprint_,tokens);return result;
    }
    std::vector<std::int64_t> suffix(int first) const {
        if(first<0 || first>position_) throw std::invalid_argument("token replay suffix extent");
        std::vector<std::int64_t> result;result.reserve(position_-first);
        const auto count=(static_cast<std::uint32_t>(position_)+63)/64;
        for(std::uint32_t index=first/64;index<count;++index) {
            const auto& page=std::get<Page>(page_node(index)->data);
            const int skip=std::max(0,first-static_cast<int>(index)*64);
            result.insert(result.end(),page.tokens.begin()+skip,page.tokens.begin()+page.rows);
        }
        return result;
    }
    // Preserve allocated_bytes as represented node storage for existing callers.
    // control_bytes adds bounded control/alignment/retirement storage; neither
    // field includes general heap allocator bookkeeping or fragmentation.
    struct Stats {
        std::uint64_t logical_tokens=0,materialized_token_records=0,allocated_bytes=0,unique_pages=0,unique_nodes=0;
        std::uint64_t control_bytes=0;
    };
    static Stats visit(std::span<const Exl3TokenHistory* const> histories,
        const std::function<void(const void*,std::size_t)>& visitor={},bool include_unused_capacity=true) {
        Stats result;std::unordered_set<const Node*> seen;
        const auto walk=[&](auto&& self,const std::shared_ptr<const Node>& node)->void {
            if(!node || !seen.insert(node.get()).second)return;
            ++result.unique_nodes;result.allocated_bytes+=sizeof(Node);
            result.control_bytes+=bounded_shared_allocation_bytes<Node>()-sizeof(Node);
            if(const auto* page=std::get_if<Page>(&node->data)) {
                ++result.unique_pages;result.materialized_token_records+=page->rows;
                if(visitor) {
                    if(include_unused_capacity)visitor(node.get(),sizeof(Node));
                    else visitor(page->tokens.data(),page->rows*sizeof(std::int64_t));
                }
            } else {
                // Payload-only visitation keeps the existing token tensor
                // contract. Full inventory above includes the radix metadata.
                if(visitor && include_unused_capacity)visitor(node.get(),sizeof(Node));
                for(const auto& child:std::get<Edges>(node->data))self(self,child);
            }
        };
        for(const auto* history:histories) {
            if(!history) throw std::invalid_argument("token history null image");
            result.logical_tokens+=history->position_;
            walk(walk,history->root_);
        }
        return result;
    }
};
} // namespace ninfer::exl3
