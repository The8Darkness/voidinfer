#pragma once
#include "exl3/activation_lifetime.h"
#include "exl3/packed_projection.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace ninfer::exl3 {
// Fixed metadata for the already-reserved Engine packed input, never a second
// allocation pool. The caller clears this before any write to that storage.
// Only a completed gather may install a signature. Owners keep witness objects
// alive; current() additionally proves the private activation has not expired.
class Exl3PackedGatherSignature {
public:
    struct Lane {
        std::uint64_t request=0,acquisition=0,execution=0;
        std::shared_ptr<const void> model,root,input_owner;
        const std::uint16_t* input=nullptr;
        int layer=-1,position=0,rows=0,columns=0;
        std::size_t stride=0;
        Exl3ActivationLifetime::Witness activation{};
        Exl3ProjectionContract contract;
        static bool same_owner(const std::shared_ptr<const void>& a,
                               const std::shared_ptr<const void>& b) noexcept {
            return a==b && !a.owner_before(b) && !b.owner_before(a);
        }
        bool valid() const noexcept {
            return request && acquisition && execution && !contract.empty() && model && root && input_owner && input &&
                model.use_count() && root.use_count() && input_owner.use_count() &&
                layer>=0 && layer<64 && position>=0 && rows>=1 && rows<=8 && columns>0 &&
                stride>=static_cast<std::size_t>(columns) && activation.current();
        }
        bool same(const Lane& other) const noexcept {
            return request==other.request && acquisition==other.acquisition && execution==other.execution &&
                same_owner(model,other.model) && same_owner(root,other.root) &&
                same_owner(input_owner,other.input_owner) && input==other.input &&
                layer==other.layer && position==other.position && rows==other.rows && columns==other.columns &&
                stride==other.stride && contract==other.contract && activation.lifetime==other.activation.lifetime &&
                activation.generation==other.activation.generation;
        }
    };
    using Pair=std::array<Lane,2>;
    void clear() noexcept {lanes_={};packed_=nullptr;elements_=0;}
    bool remember(Pair lanes,const std::uint16_t* packed,std::size_t elements) noexcept {
        clear();
        if(!packed || !lanes[0].valid() || !lanes[1].valid() ||
           lanes[0].request==lanes[1].request || !Lane::same_owner(lanes[0].model,lanes[1].model) ||
           lanes[0].contract!=lanes[1].contract || lanes[0].columns!=lanes[1].columns || lanes[0].layer!=lanes[1].layer ||
           elements/static_cast<std::size_t>(lanes[0].columns)<static_cast<std::size_t>(lanes[0].rows+lanes[1].rows))
            return false;
        lanes_=std::move(lanes);packed_=packed;elements_=elements;return true;
    }
    // One adjacent consumer only; mismatches also retire retained metadata.
    // The Engine may preserve the original physical row order when the two
    // worker threads reach the adjacent operator in the opposite order.
    int consume_order(const Pair& next,const std::uint16_t* packed,std::size_t elements) noexcept {
        int order=0;
        if(packed_ && packed_==packed && elements_==elements &&
           next[0].valid() && next[1].valid()) {
            if(lanes_[0].same(next[0]) && lanes_[1].same(next[1]))order=1;
            else if(lanes_[0].same(next[1]) && lanes_[1].same(next[0]))order=-1;
        }
        clear();return order;
    }
    bool consume(const Pair& next,const std::uint16_t* packed,std::size_t elements) noexcept {
        return consume_order(next,packed,elements)==1;
    }
private:
    Pair lanes_{};
    const std::uint16_t* packed_=nullptr;
    std::size_t elements_=0;
};
}
