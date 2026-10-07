#pragma once
// Fixed-size pinned host blocks for parked L0 OSCAR code chunks: page-locked
// so a resume uploads them with asynchronous DMA, carved from 64-block slabs
// so parking does not pay a page-locking call per chunk. Blocks return to a
// process-wide free list; slabs are kept for reuse.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace ninfer::exl3 {

class Exl3L0PinnedPool {
public:
    // One 4096-row chunk: codes 2 x 4096 x 4 x 64 B + meta 2 x 4096 x 4 x 4 x 2 B.
    static constexpr std::size_t kBlockBytes = 4096ull * 2 * 4 * 64 + 4096ull * 2 * 4 * 4 * 2;
    static constexpr std::size_t kSlabBlocks = 64;

    struct Block {
        std::uint8_t* data = nullptr;
        Block() = default;
        explicit Block(std::uint8_t* p) : data(p) {}
        Block(const Block&) = delete;
        Block& operator=(const Block&) = delete;
        Block(Block&& other) noexcept : data(other.data) { other.data = nullptr; }
        Block& operator=(Block&& other) noexcept {
            if (this != &other) { release(); data = other.data; other.data = nullptr; }
            return *this;
        }
        ~Block() { release(); }
        void release() noexcept { if (data) { instance().give_back(data); data = nullptr; } }
    };

    static Exl3L0PinnedPool& instance() { static Exl3L0PinnedPool pool; return pool; }

    Block take() {
        std::lock_guard lock(mutex_);
        if (free_.empty()) {
            void* slab = nullptr;
            if (cudaHostAlloc(&slab, kBlockBytes * kSlabBlocks, cudaHostAllocPortable) != cudaSuccess)
                throw std::runtime_error("L0 OSCAR pinned chunk slab allocation");
            slabs_.push_back(slab);
            for (std::size_t i = 0; i < kSlabBlocks; ++i)
                free_.push_back(static_cast<std::uint8_t*>(slab) + i * kBlockBytes);
        }
        auto* p = free_.back();
        free_.pop_back();
        return Block(p);
    }

private:
    void give_back(std::uint8_t* p) noexcept {
        std::lock_guard lock(mutex_);
        free_.push_back(p);
    }
    std::mutex mutex_;
    std::vector<void*> slabs_;
    std::vector<std::uint8_t*> free_;
};

}  // namespace ninfer::exl3
