#pragma once
#include <cstdint>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <array>
#include <utility>

namespace ninfer::exl3 {
// Owned, bounded contract identity. Never truncate an overlong identity or keep
// a view into caller storage: either could admit unrelated physical operations.
class Exl3ProjectionContract {
public:
    static constexpr std::size_t capacity=128;
    Exl3ProjectionContract()=default;
    Exl3ProjectionContract(std::string_view text) {
        if(text.size()>capacity)throw std::invalid_argument("packed projection contract capacity");
        size_=text.size();
        for(std::size_t i=0;i<size_;++i)bytes_[i]=text[i];
    }
    Exl3ProjectionContract(const char* text):Exl3ProjectionContract(text?std::string_view(text):std::string_view{}) {}
    Exl3ProjectionContract(const std::string& text):Exl3ProjectionContract(std::string_view(text)) {}
    bool empty() const noexcept {return size_==0;}
    friend bool operator==(const Exl3ProjectionContract& a,const Exl3ProjectionContract& b) noexcept {
        return std::string_view(a.bytes_.data(),a.size_)==std::string_view(b.bytes_.data(),b.size_);
    }
private:
    std::array<char,capacity> bytes_{};
    std::size_t size_=0;
};
// One stateless projection operation, not a queue or a second scheduler.
// Epoch liveness must be checked by the existing exclusive lease owner.
struct Exl3ProjectionRows {
    static bool same_owner(const std::shared_ptr<const void>& a,const std::shared_ptr<const void>& b) noexcept {
        return a==b && !a.owner_before(b) && !b.owner_before(a);
    }
    std::uint64_t request=0,acquisition=0,execution=0;
    std::shared_ptr<const void> model,root,input_owner,output_owner;
    Exl3ProjectionContract contract; // binds modality, represented dtype and projection role
    int position=0,rows=0,input_columns=0,output_columns=0;
    std::size_t input_stride=0,output_stride=0; // FP16 elements
    const std::uint16_t* input=nullptr;
    std::uint16_t* output=nullptr;
    // Readable/writable extents starting at the supplied pointers, established
    // by their physical owners. Rows/strides may not exceed those allocations.
    std::size_t input_storage_elements=0,output_storage_elements=0;
    bool valid_admission(int capacity) const noexcept {
        const bool geometry=request && acquisition && execution && model && root && input_owner && output_owner &&
            model.use_count() && root.use_count() && input_owner.use_count() && output_owner.use_count() &&
            !contract.empty() && input && output && rows>0 && rows<=capacity && position>=0 &&
            position<=std::numeric_limits<int>::max()-rows && input_columns>0 && output_columns>0 &&
            input_stride>=static_cast<std::size_t>(input_columns) &&
            output_stride>=static_cast<std::size_t>(output_columns) &&
            input_stride<=std::numeric_limits<std::size_t>::max()/sizeof(std::uint16_t) &&
            output_stride<=std::numeric_limits<std::size_t>::max()/sizeof(std::uint16_t);
        if(!geometry)return false;
        const auto covers=[&](std::size_t storage,std::size_t stride,int columns) {
            const auto width=static_cast<std::size_t>(columns);
            storage=std::min(storage,std::numeric_limits<std::size_t>::max()/sizeof(std::uint16_t));
            // No extent multiplication before proving the declared bound.
            return storage>=width && static_cast<std::size_t>(rows-1)<=(storage-width)/stride;
        };
        if(!covers(input_storage_elements,input_stride,input_columns) ||
            !covers(output_storage_elements,output_stride,output_columns))return false;
        const auto input_bytes=((rows-1)*input_stride+input_columns)*sizeof(std::uint16_t);
        const auto output_bytes=((rows-1)*output_stride+output_columns)*sizeof(std::uint16_t);
        const auto a=reinterpret_cast<std::uintptr_t>(input),b=reinterpret_cast<std::uintptr_t>(output);
        const auto limit=std::numeric_limits<std::uintptr_t>::max();
        if(input_bytes>limit-a || output_bytes>limit-b)return false;
        return a>=b?a-b>=output_bytes:b-a>=input_bytes;
    }
    static bool valid_pair(const Exl3ProjectionRows& a,const Exl3ProjectionRows& b,int capacity) noexcept {
        if(capacity<2 || capacity>16 || !a.valid_admission(capacity) ||
            !b.valid_admission(capacity-a.rows) || a.request==b.request ||
            !same_owner(a.model,b.model) || a.contract!=b.contract ||
            a.input_columns!=b.input_columns || a.output_columns!=b.output_columns)return false;
        // valid_admission already proved each extent and address sum cannot overflow.
        const auto bounds=[](const Exl3ProjectionRows& row,bool output) {
            const auto start=reinterpret_cast<std::uintptr_t>(output?row.output:row.input);
            const auto stride=output?row.output_stride:row.input_stride;
            const auto columns=output?row.output_columns:row.input_columns;
            return std::pair{start,start+((row.rows-1)*stride+columns)*sizeof(std::uint16_t)};
        };
        for(bool left:{false,true})for(bool right:{false,true}) {
            const auto x=bounds(a,left),y=bounds(b,right);
            if(x.first<y.second && y.first<x.second)return false;
        }
        return true;
    }
};
class Exl3PackedProjectionPlan {
public:
    using Live=std::function<bool(const Exl3ProjectionRows&)>;
    struct Lane {Exl3ProjectionRows source;int packed_first=0;};
    Exl3PackedProjectionPlan(const Exl3PackedProjectionPlan&)=default;
    Exl3PackedProjectionPlan& operator=(const Exl3PackedProjectionPlan&)=default;
    Exl3PackedProjectionPlan(Exl3PackedProjectionPlan&& other) noexcept
        :lanes_(std::move(other.lanes_)),lane_count_(std::exchange(other.lane_count_,0)),
         rows_(std::exchange(other.rows_,0)) {}
    Exl3PackedProjectionPlan& operator=(Exl3PackedProjectionPlan&& other) noexcept {
        if(this!=&other) {
            lanes_=std::move(other.lanes_);
            lane_count_=std::exchange(other.lane_count_,0);
            rows_=std::exchange(other.rows_,0);
        }
        return *this;
    }
    template<class Predicate=Live>
    static Exl3PackedProjectionPlan assemble(std::span<const Exl3ProjectionRows> sources,
                                             int capacity,const Predicate& live) {
        require_live(live);
        if(sources.size()<2 || sources.size()>8 || capacity<2 || capacity>16)
            throw std::invalid_argument("packed projection bounded membership");
        Exl3PackedProjectionPlan plan;
        for(const auto& s:sources) {
            if(!s.valid_admission(capacity-plan.rows_) || !live(s))
                throw std::invalid_argument("packed projection incompatible/stale row");
            const auto input=range(s.input,s.rows,s.input_stride,s.input_columns);
            const auto output=range(s.output,s.rows,s.output_stride,s.output_columns);
            if((input.second-input.first)/2>s.input_storage_elements ||
               (output.second-output.first)/2>s.output_storage_elements)
                throw std::invalid_argument("packed projection owner extent too small");
            if(overlap(input,output)) throw std::invalid_argument("packed projection in-place alias");
            for(const auto& lane:plan.lanes()) {
                const auto& p=lane.source;
                if(p.request==s.request || !Exl3ProjectionRows::same_owner(p.model,s.model) || p.contract!=s.contract ||
                   p.input_columns!=s.input_columns || p.output_columns!=s.output_columns ||
                   overlap(input,range(p.input,p.rows,p.input_stride,p.input_columns)) ||
                   overlap(input,range(p.output,p.rows,p.output_stride,p.output_columns)) ||
                   overlap(output,range(p.input,p.rows,p.input_stride,p.input_columns)) ||
                   overlap(output,range(p.output,p.rows,p.output_stride,p.output_columns)))
                    throw std::invalid_argument("packed projection contract/alias mismatch");
            }
            plan.lanes_[plan.lane_count_++]={s,plan.rows_};plan.rows_+=s.rows;
        }
        // Cancellation while assembling cannot leave a publishable partial plan.
        plan.validate_live(live);
        return plan;
    }
    template<class Predicate=Live>
    void validate_live(const Predicate& live) const {
        require_live(live);
        if(lane_count_<2 || rows_<2)
            throw std::invalid_argument("packed projection empty or moved-from plan");
        for(const auto& lane:lanes()) if(!live(lane.source))
            throw std::invalid_argument("packed projection canceled/stale acquisition");
    }
    std::span<const Lane> lanes() const noexcept {return {lanes_.data(),lane_count_};}
    int rows() const noexcept {return rows_;}
    void require_disjoint_scratch(const std::uint16_t* input,int input_columns,
                                  std::uint16_t* output,int output_columns) const {
        if(!input || !output || input_columns<1 || output_columns<1)
            throw std::invalid_argument("packed scratch missing");
        const auto a=range(input,rows_,input_columns,input_columns);
        const auto b=range(output,rows_,output_columns,output_columns);
        if(overlap(a,b)) throw std::invalid_argument("packed scratch alias");
        for(const auto& lane:lanes()) {
            const auto& s=lane.source;
            const auto i=range(s.input,s.rows,s.input_stride,s.input_columns);
            const auto o=range(s.output,s.rows,s.output_stride,s.output_columns);
            if(overlap(a,i) || overlap(a,o) || overlap(b,i) || overlap(b,o))
                throw std::invalid_argument("packed scratch aliases lane");
        }
    }
private:
    template<class Predicate>
    static void require_live(const Predicate& live) {
        // Preserve empty std::function/function-pointer refusal while allowing
        // actual Engine closures to pass by reference without type erasure.
        if constexpr(requires {static_cast<bool>(live);})
            if(!static_cast<bool>(live))throw std::invalid_argument("packed projection liveness owner missing");
    }
    using Range=std::pair<std::uintptr_t,std::uintptr_t>;
    static Range range(const void* pointer,int rows,std::size_t stride,int columns) {
        const auto limit=std::numeric_limits<std::size_t>::max()/2;
        if(stride>limit || static_cast<std::size_t>(columns)>limit ||
           static_cast<std::size_t>(rows-1)>(limit-columns)/stride)
            throw std::overflow_error("packed projection byte extent");
        const auto bytes=((rows-1)*stride+columns)*2;
        const auto start=reinterpret_cast<std::uintptr_t>(pointer);
        if(bytes>std::numeric_limits<std::uintptr_t>::max()-start)
            throw std::overflow_error("packed projection address extent");
        return {start,start+bytes};
    }
    static bool overlap(Range a,Range b) noexcept {return a.first<b.second && b.first<a.second;}
    Exl3PackedProjectionPlan()=default;
    // Membership is bounded at eight by admission. Keep its table inside the
    // retained plan so Engine startup metadata accounts it before dispatch.
    // Each source's contract identity is also stored inline.
    std::array<Lane,8> lanes_{};
    std::size_t lane_count_=0;
    int rows_=0;
};
} // namespace ninfer::exl3
