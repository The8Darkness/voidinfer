#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

namespace ninfer::exl3 {
enum class Exl3GraphPointerRole : std::uint8_t {
    token_ids=0,hidden_output=1,embedding_trace=2,count=3
};
struct Exl3GraphRoleBinding {
    Exl3GraphPointerRole role=Exl3GraphPointerRole::count;
    std::shared_ptr<const void> owner;
    void* address=nullptr;
    std::size_t elements=0,element_bytes=0;
    bool writable=false;
};
struct Exl3DeviceGraphRoleRecord {
    const std::int64_t* token_ids=nullptr;
    std::uint16_t* hidden_output=nullptr;
    std::uint16_t* embedding_trace=nullptr;
    std::uint32_t rows=0,hidden_width=0;
    std::uint64_t generation=0;
};
struct Exl3GraphRoleUpdate {
    Exl3DeviceGraphRoleRecord record;
    bool changed=false;
};

// Host authority for one stable device record read by a captured kernel. Every
// role is replaced as one generation; no partial table state is observable.
class Exl3DeviceGraphRoleTable {
public:
    static constexpr std::size_t role_count=static_cast<std::size_t>(
        Exl3GraphPointerRole::count);
    void configure(unsigned rows,unsigned hidden_width) {
        if(configured_ || !rows || rows>16 || !hidden_width)
            throw std::logic_error("graph role table configuration");
        rows_=rows;hidden_width_=hidden_width;configured_=true;
    }
    Exl3GraphRoleUpdate update(std::span<const Exl3GraphRoleBinding> values) {
        if(!configured_ || values.size()!=role_count)
            throw std::invalid_argument("graph role table partial update");
        std::array<Exl3GraphRoleBinding,role_count> next{};
        std::array<bool,role_count> seen{};
        for(const auto& value:values) {
            const auto slot=static_cast<std::size_t>(value.role);
            if(slot>=role_count || seen[slot])
                throw std::invalid_argument("graph role table reused role slot");
            seen[slot]=true;
            const bool token=slot==static_cast<std::size_t>(
                Exl3GraphPointerRole::token_ids);
            const auto expected_elements=token?static_cast<std::size_t>(rows_):
                static_cast<std::size_t>(rows_)*hidden_width_;
            const auto expected_bytes=token?sizeof(std::int64_t):sizeof(std::uint16_t);
            if(!value.owner || !value.address || value.elements!=expected_elements ||
               value.element_bytes!=expected_bytes || value.writable==token ||
               reinterpret_cast<std::uintptr_t>(value.address)%expected_bytes ||
               !valid_extent(value))
                throw std::invalid_argument("graph role table incompatible pointer geometry");
            next[slot]=value;
        }
        for(std::size_t i=0;i<role_count;++i)for(std::size_t j=0;j<i;++j)
            if(overlaps(next[i],next[j]) && (next[i].writable || next[j].writable))
                throw std::invalid_argument("graph role table aliases writable extent");
        bool changed=!generation_;
        for(std::size_t i=0;i<role_count && !changed;++i)
            changed=!same_binding(bindings_[i],next[i]);
        if(!changed)return {record_,false};
        if(pending_replay_)
            throw std::logic_error("graph role table update before final use");
        if(generation_==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("graph role table generation exhausted");
        bindings_=std::move(next);uploaded_generation_=0;++generation_;
        record_.token_ids=static_cast<const std::int64_t*>(bindings_[0].address);
        record_.hidden_output=static_cast<std::uint16_t*>(bindings_[1].address);
        record_.embedding_trace=static_cast<std::uint16_t*>(bindings_[2].address);
        record_.rows=rows_;record_.hidden_width=hidden_width_;
        record_.generation=generation_;
        return {record_,true};
    }
    void mark_uploaded(std::uint64_t generation) {
        if(!generation || generation!=generation_ || pending_replay_)
            throw std::logic_error("graph role table upload generation");
        uploaded_generation_=generation;
    }
    void begin_use(std::uint64_t generation,std::uint64_t replay_serial) {
        if(!generation || generation!=generation_ || generation!=uploaded_generation_ ||
           !replay_serial || (pending_replay_ && replay_serial<=pending_replay_))
            throw std::logic_error("graph role table stale generation/replay");
        pending_replay_=replay_serial;
    }
    bool complete_use(std::uint64_t replay_serial) noexcept {
        if(!pending_replay_ || replay_serial!=pending_replay_)return false;
        pending_replay_=0;return true;
    }
    std::uint64_t generation() const noexcept {return generation_;}
    std::uint64_t uploaded_generation() const noexcept {return uploaded_generation_;}
    std::uint64_t pending_replay() const noexcept {return pending_replay_;}
    const Exl3DeviceGraphRoleRecord& record() const noexcept {return record_;}
private:
    static bool same_owner(const std::shared_ptr<const void>& a,
        const std::shared_ptr<const void>& b) noexcept {
        return !a.owner_before(b) && !b.owner_before(a);
    }
    static bool same_binding(const Exl3GraphRoleBinding& a,
        const Exl3GraphRoleBinding& b) noexcept {
        return a.role==b.role && same_owner(a.owner,b.owner) &&
            a.address==b.address && a.elements==b.elements &&
            a.element_bytes==b.element_bytes && a.writable==b.writable;
    }
    static bool valid_extent(const Exl3GraphRoleBinding& value) noexcept {
        if(value.elements>std::numeric_limits<std::size_t>::max()/value.element_bytes)
            return false;
        const auto bytes=value.elements*value.element_bytes;
        const auto first=reinterpret_cast<std::uintptr_t>(value.address);
        return first<=std::numeric_limits<std::uintptr_t>::max()-bytes;
    }
    static bool overlaps(const Exl3GraphRoleBinding& a,
        const Exl3GraphRoleBinding& b) noexcept {
        const auto first=reinterpret_cast<std::uintptr_t>(a.address);
        const auto other=reinterpret_cast<std::uintptr_t>(b.address);
        const auto bytes=a.elements*a.element_bytes;
        const auto other_bytes=b.elements*b.element_bytes;
        return first<other+other_bytes && other<first+bytes;
    }
    std::array<Exl3GraphRoleBinding,role_count> bindings_{};
    Exl3DeviceGraphRoleRecord record_{};
    unsigned rows_=0,hidden_width_=0;
    std::uint64_t generation_=0,uploaded_generation_=0,pending_replay_=0;
    bool configured_=false;
};
}
