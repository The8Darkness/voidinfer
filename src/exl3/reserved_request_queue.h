#pragma once
#include <memory>
#include <stdexcept>
#include <vector>

namespace ninfer::exl3 {
template<class Request> class ReservedRequestQueue {
public:
    using Entry=std::shared_ptr<Request>;
    class Storage {
        friend class ReservedRequestQueue;
        std::vector<Entry> entries;
        bool bound_=false;
    public:
        Storage()=default;
        Storage(const Storage&)=delete;
        Storage& operator=(const Storage&)=delete;
        void reserve(std::size_t slots) {
            if(bound_)throw std::logic_error("request queue storage already bound");
            entries.reserve(slots);
        }
        std::size_t capacity() const noexcept {return entries.capacity();}
    };
private:
    std::shared_ptr<Storage> owner_;
    std::vector<Entry> empty_; // Empty startup/failed-construction iteration.
    auto& entries() noexcept {return owner_?owner_->entries:empty_;}
public:
    ReservedRequestQueue()=default;
    ReservedRequestQueue(const ReservedRequestQueue&)=delete;
    ReservedRequestQueue& operator=(const ReservedRequestQueue&)=delete;
    void bind(std::shared_ptr<Storage> owner) {
        if(owner_ || !owner || owner->bound_ || !owner->entries.empty() || !owner->entries.capacity())
            throw std::logic_error("request queue storage binding");
        owner->bound_=true;
        owner_=std::move(owner);
    }
    auto begin() noexcept {return entries().begin();}
    auto end() noexcept {return entries().end();}
    bool empty() noexcept {return entries().empty();}
    std::size_t size() noexcept {return entries().size();}
    std::size_t capacity() noexcept {return entries().capacity();}
    Entry& front() {return entries().front();}
    void push_back(const Entry& value) {
        if(!owner_ || entries().size()==entries().capacity())
            throw std::length_error("request queue reserved capacity exhausted");
        entries().push_back(value);
    }
    auto erase(typename std::vector<Entry>::iterator position) {return entries().erase(position);}
    void pop_front() {entries().erase(entries().begin());}
};
}
