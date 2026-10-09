#pragma once
// Lane-backed exact KV pages. A context exporting its exact state keeps the
// L0 FP16 history in its own planes (mapped host memory) anyway, so the
// exported 64-row pages hold no payload copy: they are `deferred` and name
// the context's source. The source fills them from the planes only when the
// rows they cover are about to change (lineage reset, rewind, restore of
// another history, context destruction) or when another reader needs the
// values (restore into another lane, comparisons, diagnostics). Restoring a
// still-deferred page into its own lane copies nothing.
//
// Rows below an exported frontier are append-only history: every operation
// that rewrites such rows calls materialize_from/materialize_except first.
// Fills run on the source's own stream; the planes' rows they read are not
// written concurrently (the lane only appends beyond them).
#include "exl3/exact_kv_extent.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <execution>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::exl3 {

class Exl3LaneKvSource : public std::enable_shared_from_this<Exl3LaneKvSource> {
public:
    static constexpr int kRowWords = 1024;          // FP16 [4][256] per row and bank
    static constexpr int kStageRows = 8192;
    struct Planes {
        std::array<const std::uint16_t*, 16> k{}, v{};
        int banks = 0;
        bool cpu_visible = false;
    };
    explicit Exl3LaneKvSource(const Planes& planes) : planes_(planes) {}
    ~Exl3LaneKvSource() {
        if (stream_) cudaStreamDestroy(stream_);
        if (stage_) cudaFreeHost(stage_);
    }
    Exl3LaneKvSource(const Exl3LaneKvSource&) = delete;
    Exl3LaneKvSource& operator=(const Exl3LaneKvSource&) = delete;

    std::mutex& mutex() noexcept { return mutex_; }

    // Publishes `page` (rows [first, first+rows) of these planes) as deferred.
    void defer(const std::shared_ptr<Exl3ExactKVPage>& page) {
        std::lock_guard lock(mutex_);
        for (int bank = 0; bank < 16; ++bank) {
            std::vector<std::uint16_t>().swap(page->k[bank]);
            std::vector<std::uint16_t>().swap(page->v[bank]);
        }
        page->source = shared_from_this();
        page->deferred.store(true);
        if (pages_.size() >= 2 * live_hint_ + 256) prune_locked();
        pages_.push_back(page);
    }

    // Fills one page (another reader needs its values).
    void materialize(const Exl3ExactKVPage& page) {
        std::lock_guard lock(mutex_);
        if (!page.payload_deferred()) return;
        std::vector<Exl3ExactKVPage*> one{const_cast<Exl3ExactKVPage*>(&page)};
        fill_locked(one);
    }

    // Fills every deferred page holding a row >= `row` (rows about to change).
    void materialize_from(int row) {
        std::lock_guard lock(mutex_);
        std::vector<std::shared_ptr<Exl3ExactKVPage>> owners;
        std::vector<Exl3ExactKVPage*> pages;
        for (const auto& weak : pages_)
            if (auto page = weak.lock())
                if (page->payload_deferred() && page->first + page->rows > row) {
                    pages.push_back(page.get());
                    owners.push_back(std::move(page));
                }
        fill_locked(pages);
        prune_locked();
    }

    // Fills every deferred page except `keep` (a restore rewrites the planes
    // with `keep`'s history, which these planes already hold).
    void materialize_except(const std::vector<std::shared_ptr<const Exl3ExactKVPage>>& keep) {
        std::lock_guard lock(mutex_);
        std::vector<const Exl3ExactKVPage*> kept;
        kept.reserve(keep.size());
        for (const auto& page : keep) kept.push_back(page.get());
        std::sort(kept.begin(), kept.end());
        std::vector<std::shared_ptr<Exl3ExactKVPage>> owners;
        std::vector<Exl3ExactKVPage*> pages;
        for (const auto& weak : pages_)
            if (auto page = weak.lock())
                if (page->payload_deferred() &&
                    !std::binary_search(kept.begin(), kept.end(), static_cast<const Exl3ExactKVPage*>(page.get()))) {
                    pages.push_back(page.get());
                    owners.push_back(std::move(page));
                }
        fill_locked(pages);
        prune_locked();
    }

    // Some still-deferred page holds a row >= `row`.
    bool deferred_beyond(int row) {
        std::lock_guard lock(mutex_);
        for (const auto& weak : pages_)
            if (auto page = weak.lock())
                if (page->payload_deferred() && page->first + page->rows > row) return true;
        return false;
    }

    // Copies the rows of a contiguous run of this source's still-deferred
    // pages straight into another lane's planes (a restore there), without
    // filling them. False if a page was filled meanwhile (use its values).
    bool copy_run_to(const std::vector<const Exl3ExactKVPage*>& run, const Planes& destination,
                     cudaStream_t stream) {
        std::lock_guard lock(mutex_);
        for (const auto* page : run)
            if (!page->payload_deferred()) return false;
        if (run.empty()) return true;
        const int lo = run.front()->first, hi = run.back()->first + run.back()->rows;
        const std::size_t offset = static_cast<std::size_t>(lo) * kRowWords;
        const std::size_t bytes = static_cast<std::size_t>(hi - lo) * kRowWords * sizeof(std::uint16_t);
        for (int bank = 0; bank < planes_.banks; ++bank) {
            check(cudaMemcpyAsync(const_cast<std::uint16_t*>(destination.k[bank]) + offset, planes_.k[bank] + offset,
                bytes, cudaMemcpyDefault, stream), "lane KV plane copy");
            check(cudaMemcpyAsync(const_cast<std::uint16_t*>(destination.v[bank]) + offset, planes_.v[bank] + offset,
                bytes, cudaMemcpyDefault, stream), "lane KV plane copy");
        }
        // The rows stay readable only while this lock is held.
        check(cudaStreamSynchronize(stream), "lane KV plane copy");
        return true;
    }
    const Planes& planes() const noexcept { return planes_; }

private:
    void prune_locked() {
        pages_.erase(std::remove_if(pages_.begin(), pages_.end(), [](const auto& weak) {
            const auto page = weak.lock();
            return !page || !page->payload_deferred();
        }), pages_.end());
        live_hint_ = pages_.size();
    }

    static void check(cudaError_t error, const char* what) {
        if (error != cudaSuccess)
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }

    void fill_locked(std::vector<Exl3ExactKVPage*>& pages) {
        if (pages.empty()) return;
        std::sort(pages.begin(), pages.end(), [](const auto* a, const auto* b) { return a->first < b->first; });
        for (auto* page : pages)
            for (int bank = 0; bank < planes_.banks; ++bank) {
                page->k[bank].resize(static_cast<std::size_t>(page->rows) * kRowWords);
                page->v[bank].resize(static_cast<std::size_t>(page->rows) * kRowWords);
            }
        if (planes_.cpu_visible) {
            std::for_each(std::execution::par, pages.begin(), pages.end(), [&](Exl3ExactKVPage* page) {
                for (int bank = 0; bank < planes_.banks; ++bank) {
                    const std::size_t offset = static_cast<std::size_t>(page->first) * kRowWords;
                    const std::size_t bytes = static_cast<std::size_t>(page->rows) * kRowWords * sizeof(std::uint16_t);
                    std::memcpy(page->k[bank].data(), planes_.k[bank] + offset, bytes);
                    std::memcpy(page->v[bank].data(), planes_.v[bank] + offset, bytes);
                }
            });
        } else {
            if (!stream_) check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "lane KV source stream");
            if (!stage_)
                check(cudaMallocHost(reinterpret_cast<void**>(&stage_),
                    static_cast<std::size_t>(kStageRows) * kRowWords * sizeof(std::uint16_t)), "lane KV source stage");
            // Contiguous row runs of the (sorted) pages, in stage-sized windows.
            std::size_t begin = 0;
            while (begin < pages.size()) {
                std::size_t end = begin + 1;
                int lo = pages[begin]->first, hi = pages[begin]->first + pages[begin]->rows;
                while (end < pages.size() && pages[end]->first == hi &&
                       hi + pages[end]->rows - lo <= kStageRows) {
                    hi += pages[end]->rows;
                    ++end;
                }
                for (int bank = 0; bank < planes_.banks; ++bank)
                    for (int plane = 0; plane < 2; ++plane) {
                        const auto* source = (plane ? planes_.v[bank] : planes_.k[bank]) +
                            static_cast<std::size_t>(lo) * kRowWords;
                        check(cudaMemcpyAsync(stage_, source,
                            static_cast<std::size_t>(hi - lo) * kRowWords * sizeof(std::uint16_t),
                            cudaMemcpyDefault, stream_), "lane KV source fill");
                        check(cudaStreamSynchronize(stream_), "lane KV source fill");
                        std::for_each(std::execution::par, pages.begin() + begin, pages.begin() + end,
                            [&](Exl3ExactKVPage* page) {
                                auto& out = plane ? page->v[bank] : page->k[bank];
                                std::memcpy(out.data(), stage_ + static_cast<std::size_t>(page->first - lo) * kRowWords,
                                    static_cast<std::size_t>(page->rows) * kRowWords * sizeof(std::uint16_t));
                            });
                    }
                begin = end;
            }
        }
        for (auto* page : pages) page->deferred.store(false);
    }

    Planes planes_;
    std::mutex mutex_;
    std::vector<std::weak_ptr<Exl3ExactKVPage>> pages_;
    std::size_t live_hint_ = 0;
    cudaStream_t stream_ = nullptr;
    std::uint16_t* stage_ = nullptr;
};

// Fills a page whose rows live in another lane's planes (or its own) before
// its values are read.
inline void exl3_require_kv_payload(const Exl3ExactKVPage& page) {
    if (page.payload_deferred() && page.source) page.source->materialize(page);
}

}  // namespace ninfer::exl3
