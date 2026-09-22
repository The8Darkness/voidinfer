#pragma once
#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace ninfer::exl3 {
// Fixed owner slots are part of the enclosing object's metadata credit. New
// allocation families must explicitly fit the menu before physical allocation.
template<class Owner,std::size_t Capacity>
class Exl3FixedAllocationOwners {
public:
    std::size_t size() const noexcept {return size_;}
    constexpr std::size_t capacity() const noexcept {return Capacity;}
    constexpr std::size_t max_size() const noexcept {return Capacity;}
    template<class Visitor> void visit_owners(Visitor&& visitor) const {
        for(std::size_t i=0;i<size_;++i)visitor(owners_[i].get());
    }
    void reserve(std::size_t count) const {
        if(count>Capacity)throw std::length_error("fixed allocation owner menu exhausted");
    }
    void push_back(std::unique_ptr<Owner> owner) {
        if(size_==Capacity)throw std::length_error("fixed allocation owner menu exhausted");
        owners_[size_++]=std::move(owner);
    }
private:
    std::array<std::unique_ptr<Owner>,Capacity> owners_{};
    std::size_t size_=0;
};
}
