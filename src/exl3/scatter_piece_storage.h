#pragma once
#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ninfer::exl3 {
// Inline in the credited context owner. Clearing releases retained page owners;
// merely resetting the logical size would retain pages across slot reuse.
template<class T,std::size_t Capacity> class Exl3ScatterPieceStorage {
    static_assert(Capacity>0 && std::is_nothrow_default_constructible_v<T> &&
        std::is_nothrow_move_assignable_v<T>);
    std::array<T,Capacity> pieces_{};
    std::size_t size_=0;
public:
    Exl3ScatterPieceStorage()=default;
    Exl3ScatterPieceStorage(const Exl3ScatterPieceStorage&)=delete;
    Exl3ScatterPieceStorage& operator=(const Exl3ScatterPieceStorage&)=delete;
    Exl3ScatterPieceStorage(Exl3ScatterPieceStorage&&)=delete;
    Exl3ScatterPieceStorage& operator=(Exl3ScatterPieceStorage&&)=delete;
    void push_back(T piece) {
        if(size_==Capacity)throw std::length_error("pinned scatter piece capacity exhausted");
        pieces_[size_++]=std::move(piece);
    }
    void clear() noexcept {
        while(size_)pieces_[--size_]=T{};
    }
    bool empty() const noexcept {return size_==0;}
    std::size_t size() const noexcept {return size_;}
    static constexpr std::size_t capacity() noexcept {return Capacity;}
    const T* begin() const noexcept {return pieces_.data();}
    const T* end() const noexcept {return pieces_.data()+size_;}
};
}
