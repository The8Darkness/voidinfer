#pragma once
#include <stdexcept>

namespace ninfer::exl3 {
// Synchronous borrow only. The containing model must remain alive and its
// collection must not move/mutate. This is not an asynchronous retention token.
template<class Descriptor> class Exl3BorrowedDescriptor {
    const void* owner_;
    const Descriptor* descriptor_;
public:
    Exl3BorrowedDescriptor(const void* owner,const Descriptor& descriptor)
        :owner_(owner),descriptor_(&descriptor) {
        if(!owner_)throw std::invalid_argument("descriptor requires live collection owner");
    }
    const Descriptor& get(const void* owner) const {
        if(!owner || owner!=owner_)
            throw std::invalid_argument("descriptor collection owner mismatch");
        return *descriptor_;
    }
};
} // namespace ninfer::exl3
