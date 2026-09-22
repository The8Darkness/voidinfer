#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include <memory>
#include <utility>
#include <array>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::exl3 {
// Dispatch-time projection options must match the captured arithmetic route.
// Fixed storage avoids metadata allocation at graph admission. Oversized values
// are unrepresentable and cannot authorize graph reuse (no truncated identity).
struct Exl3ProjectionGraphOptions {
    std::array<std::array<char,64>,4> values{};
    std::array<bool,4> present{};
    bool valid=true;
    static Exl3ProjectionGraphOptions from_values(std::array<const char*,4> input) noexcept {
        Exl3ProjectionGraphOptions result;
        for(std::size_t i=0;i<input.size();++i) {
            if(!input[i])continue;
            result.present[i]=true;
            std::size_t n=0;
            for(;n<result.values[i].size();++n) {
                if(!input[i][n])break;
                result.values[i][n]=input[i][n];
            }
            if(n==result.values[i].size())result.valid=false;
        }
        return result;
    }
    static Exl3ProjectionGraphOptions current() noexcept {
        return from_values({std::getenv("NINFER_EXL3_GENERIC_SPLITS"),
            std::getenv("NINFER_EXL3_LARGE_DOWN_SPLITS"),
            std::getenv("NINFER_EXL3_LARGE_DOWN_TOPOLOGY"),
            std::getenv("NINFER_EXL3_REUSE_EQUAL_INPUT_TRANSFORMS")});
    }
    bool matches(const Exl3ProjectionGraphOptions& other) const noexcept {
        return valid && other.valid && present==other.present && values==other.values;
    }
};
// Context-stable projection configuration lives in these immutable backing
// groups. Retain both until the graph entry is retired; a raw address alone
// cannot distinguish replacement with another shared ownership control block.
class Exl3ProjectionGraphBinding {
public:
    void bind(std::shared_ptr<const void> model,std::shared_ptr<const void> scratch,
              Exl3ProjectionGraphOptions options=Exl3ProjectionGraphOptions::current()) noexcept {
        model_=std::move(model);scratch_=std::move(scratch);
        options_=options;
        // A non-null alias of an empty shared_ptr retains no backing storage.
        // Scratch may be absent for direct projections, but an alias-null owner
        // is not the same declaration as an absent scratch group.
        valid_=model_ && model_.use_count()!=0 &&
            (scratch_?scratch_.use_count()!=0:scratch_.use_count()==0) && options_.valid;
    }
    // Revoking admission must not release owners of an executable that may
    // still exist or have pending work. Recapture/destruction handles retirement.
    void invalidate() noexcept {valid_=false;}
    bool matches(const std::shared_ptr<const void>& model,
                 const std::shared_ptr<const void>& scratch,
                 Exl3ProjectionGraphOptions options=Exl3ProjectionGraphOptions::current()) const noexcept {
        return valid_ && model_ && same(model_,model) && same(scratch_,scratch) && options_.matches(options);
    }
private:
    static bool same(const std::shared_ptr<const void>& a,
                     const std::shared_ptr<const void>& b) noexcept {
        return a.get()==b.get() && !a.owner_before(b) && !b.owner_before(a);
    }
    std::shared_ptr<const void> model_,scratch_;
    bool valid_=false;
    Exl3ProjectionGraphOptions options_;
};

enum class Exl3GraphPrecision : std::uint8_t {
    oscar_int2_fp16 = 1
};
enum class Exl3GraphPositionPolicy : std::uint8_t {
    oscar_split_class = 1
};
struct Exl3GraphBufferIdentity {
    const void* address=nullptr;
    std::size_t bytes=0;
    bool operator==(const Exl3GraphBufferIdentity&) const noexcept = default;
};

// Complete immutable replay key for the context-stable numerical graph slice.
// Fixed storage keeps comparison allocation-free at dispatch.
class Exl3GraphCompatibilityFingerprint {
public:
    static constexpr std::size_t buffer_capacity=16;
    void bind(std::shared_ptr<const void> context_identity,
        std::shared_ptr<const void> model,std::shared_ptr<const void> scratch,
        std::span<const Exl3GraphBufferIdentity> buffers,unsigned rows,
        unsigned capacity,unsigned native_width,unsigned hidden_width,
        int split_class,std::uint32_t route_bits,Exl3GraphPrecision precision,
        Exl3GraphPositionPolicy position_policy,std::uint64_t generation,
        cudaStream_t stream,Exl3ProjectionGraphOptions options=
            Exl3ProjectionGraphOptions::current()) noexcept {
        valid_=false;buffer_count_=0;
        if(!context_identity || !model || buffers.empty() ||
           buffers.size()>buffer_capacity || !rows || rows>capacity ||
           !native_width || !hidden_width || split_class<1 || !route_bits ||
           !generation || !options.valid)return;
        for(const auto& buffer:buffers)
            if(!buffer.address || !buffer.bytes)return;
        context_identity_=std::move(context_identity);model_=std::move(model);
        scratch_=std::move(scratch);options_=options;
        std::copy(buffers.begin(),buffers.end(),buffers_.begin());
        buffer_count_=buffers.size();rows_=rows;capacity_=capacity;
        native_width_=native_width;hidden_width_=hidden_width;
        split_class_=split_class;route_bits_=route_bits;precision_=precision;
        position_policy_=position_policy;generation_=generation;stream_=stream;
        valid_=true;
    }
    bool matches(const Exl3GraphCompatibilityFingerprint& other) const noexcept {
        if(!valid_ || !other.valid_ || !same(context_identity_,other.context_identity_) ||
           !same(model_,other.model_) || !same(scratch_,other.scratch_) ||
           !options_.matches(other.options_) || buffer_count_!=other.buffer_count_ ||
           rows_!=other.rows_ || capacity_!=other.capacity_ ||
           native_width_!=other.native_width_ || hidden_width_!=other.hidden_width_ ||
           split_class_!=other.split_class_ || route_bits_!=other.route_bits_ ||
           precision_!=other.precision_ || position_policy_!=other.position_policy_ ||
           generation_!=other.generation_ || stream_!=other.stream_)return false;
        return std::equal(buffers_.begin(),buffers_.begin()+buffer_count_,
            other.buffers_.begin());
    }
    bool valid() const noexcept {return valid_;}
    void invalidate() noexcept {valid_=false;}
private:
    static bool same(const std::shared_ptr<const void>& a,
                     const std::shared_ptr<const void>& b) noexcept {
        return a.get()==b.get() && !a.owner_before(b) && !b.owner_before(a);
    }
    std::shared_ptr<const void> context_identity_,model_,scratch_;
    std::array<Exl3GraphBufferIdentity,buffer_capacity> buffers_{};
    std::size_t buffer_count_=0;
    unsigned rows_=0,capacity_=0,native_width_=0,hidden_width_=0;
    int split_class_=0;
    std::uint32_t route_bits_=0;
    Exl3GraphPrecision precision_=Exl3GraphPrecision::oscar_int2_fp16;
    Exl3GraphPositionPolicy position_policy_=Exl3GraphPositionPolicy::oscar_split_class;
    std::uint64_t generation_=0;
    cudaStream_t stream_=nullptr;
    Exl3ProjectionGraphOptions options_;
    bool valid_=false;
};
}
