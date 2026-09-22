#pragma once
#include "ninfer/types.h"
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace ninfer::exl3 {
class EngineRoundTokenStorage {
    // Fixed request-owned capacity includes two dependent B8 blocks. This
    // storage bound does not change the legal per-call verifier horizon.
    std::array<TokenId,16> output_{};
    std::array<std::int64_t,16> repair_{};
    std::size_t count_=0;
    // Epoch one represents the initial empty arena. Every logical rewrite,
    // including truncation, invalidates all earlier borrowed views.
    std::uint64_t epoch_=1;
    std::uint64_t next_epoch() const {
        if(epoch_==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Engine round token epoch exhausted");
        return epoch_+1;
    }
public:
    template<class T> class EpochView {
        const EngineRoundTokenStorage* owner_=nullptr;
        const T* data_=nullptr;
        std::size_t count_=0;
        std::uint64_t epoch_=0;
        friend class EngineRoundTokenStorage;
        EpochView(const EngineRoundTokenStorage* owner,const T* data,
            std::size_t count,std::uint64_t epoch) noexcept
            :owner_(owner),data_(data),count_(count),epoch_(epoch) {}
        void require_current() const {
            if(!owner_ || !epoch_ || owner_->epoch_!=epoch_)
                throw std::logic_error("stale Engine round token view");
        }
    public:
        using value_type=T;
        using const_iterator=const T*;
        EpochView()=default;
        std::size_t size() const {require_current();return count_;}
        bool empty() const {return size()==0;}
        const T* data() const {require_current();return data_;}
        const T& operator[](std::size_t index) const {
            require_current();
            if(index>=count_)throw std::out_of_range("Engine round token view index");
            return data_[index];
        }
        const T& front() const {return (*this)[0];}
        const T& back() const {
            require_current();
            if(!count_)throw std::out_of_range("Engine round token view empty");
            return data_[count_-1];
        }
        const_iterator begin() const {return data();}
        const_iterator end() const {require_current();return data_+count_;}
        std::span<const T> span() const {return {data(),count_};}
        operator std::span<const T>() const {return span();}
    };
    using TokenView=EpochView<TokenId>;
    using RepairView=EpochView<std::int64_t>;

    TokenView assign(std::span<const std::int64_t> tokens) {
        if(tokens.size()>output_.size())throw std::invalid_argument("Engine round exceeds sixteen-token storage");
        for(const auto token:tokens)
            if(token<0 || token>std::numeric_limits<TokenId>::max())
                throw std::invalid_argument("Engine round token outside TokenId range");
        const auto epoch=next_epoch();
        for(std::size_t i=0;i<tokens.size();++i) {
            output_[i]=static_cast<TokenId>(tokens[i]);repair_[i]=tokens[i];
        }
        count_=tokens.size();epoch_=epoch;return output();
    }
    TokenView truncate(std::size_t count) {
        if(count>count_)throw std::invalid_argument("Engine decoder expanded committed token prefix");
        const auto epoch=next_epoch();
        count_=count;epoch_=epoch;return output();
    }
    TokenView output() const noexcept {return {this,output_.data(),count_,epoch_};}
    RepairView repair() const noexcept {return {this,repair_.data(),count_,epoch_};}
};
}
