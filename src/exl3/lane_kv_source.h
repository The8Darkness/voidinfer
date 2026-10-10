#pragma once
// Lane-backed exact KV pages. A context exporting its exact state keeps the
// L0 FP16 history in its own planes (mapped host memory) anyway, so the
// exported 64-row pages hold no payload copy: they are `deferred` and name
// the context's source. The source fills them from the planes only when the
// rows they cover are about to change (lineage reset, rewind, restore of
// another history, context destruction) or when another reader needs the
// values. Restoring a still-deferred page into its own lane copies nothing;
// restoring a filled page into a lane re-homes it there (adopt), so every
// history row has one host copy.
//
// Rows below an exported frontier are append-only history: every operation
// that rewrites such rows fills the affected pages first.
//
// Under host memory pressure (kv_spill_store.h) a fill writes the rows to the
// spill file instead (the page stays deferred, now `spilled`), and filled
// pages of every source are spilled before the planes grow. A restore copies
// spilled rows straight into its planes and re-homes the pages; other readers
// bring them back into RAM.
//
// Locking: a page's owner (`source`), payload vectors and `deferred` flag
// change only under the process-wide payload mutex held exclusively; readers
// of payload vectors hold it shared. `source_id` and `deferred` are atomics
// for lock-free identity checks. A source's own mutex guards its page list
// and is always taken before the payload mutex.
#include "exl3/exact_kv_extent.h"
#include "exl3/kv_spill_store.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <execution>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::exl3 {

inline std::shared_mutex& exl3_page_payload_mutex() {
    static std::shared_mutex mutex;
    return mutex;
}

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
    static std::shared_ptr<Exl3LaneKvSource> create(const Planes& planes) {
        auto source = std::make_shared<Exl3LaneKvSource>(planes);
        std::lock_guard lock(registry_mutex());
        auto& sources = registry();
        sources.erase(std::remove_if(sources.begin(), sources.end(),
            [](const auto& weak) { return weak.expired(); }), sources.end());
        sources.push_back(source);
        return source;
    }
    ~Exl3LaneKvSource() {
        if (stream_) cudaStreamDestroy(stream_);
        if (stage_) cudaFreeHost(stage_);
        if (swap_stage_) cudaFreeHost(swap_stage_);
    }
    Exl3LaneKvSource(const Exl3LaneKvSource&) = delete;
    Exl3LaneKvSource& operator=(const Exl3LaneKvSource&) = delete;

    const Planes& planes() const noexcept { return planes_; }

    // The page's rows still live only in this source's planes.
    bool holds(const Exl3ExactKVPage& page) const noexcept {
        return page.source_id.load() == this && page.payload_deferred() && !page.spilled();
    }
    // The page's rows live only in this source's spill slot.
    bool spilled_here(const Exl3ExactKVPage& page) const noexcept {
        return page.source_id.load() == this && page.payload_deferred() && page.spilled();
    }

    // Publishes `page` (rows [first, first+rows) of these planes) as deferred.
    void defer(const std::shared_ptr<Exl3ExactKVPage>& page) {
        std::lock_guard lock(mutex_);
        std::unique_lock payload(exl3_page_payload_mutex());
        release_payload(*page);
        page->spill.reset();
        page->source = shared_from_this();
        page->source_id.store(this);
        page->deferred.store(true);
        track_locked(page);
    }

    // Re-homes filled or spilled lane-backed pages whose rows were just
    // restored into these planes: their payload copies / slots are released.
    void adopt(const std::vector<std::shared_ptr<const Exl3ExactKVPage>>& pages) {
        std::lock_guard lock(mutex_);
        std::unique_lock payload(exl3_page_payload_mutex());
        for (const auto& constant : pages) {
            auto page = std::const_pointer_cast<Exl3ExactKVPage>(constant);
            if (!page->source || (page->payload_deferred() && !page->spilled())) continue;
            // Another restore still reads this page's payload.
            if (!page->payload_deferred() && page->pins.load() > 1) continue;
            release_payload(*page);
            page->spill.reset();
            page->source = shared_from_this();
            page->source_id.store(this);
            page->deferred.store(true);
            track_locked(page);
        }
    }

    // Fills one of this source's pages (another reader needs its values).
    void materialize(const Exl3ExactKVPage& page) {
        std::lock_guard lock(mutex_);
        std::unique_lock payload(exl3_page_payload_mutex());
        auto& mutable_page = const_cast<Exl3ExactKVPage&>(page);
        if (spilled_here(page)) {
            unspill_locked(mutable_page);
            return;
        }
        if (!holds(page)) return;
        std::vector<Exl3ExactKVPage*> one{&mutable_page};
        fill_locked(one, false);
    }

    // Host memory is short: spills filled lane-backed pages of every source
    // (idle histories) until `extra` more bytes fit above the spill floor.
    static void relieve_memory_pressure(std::uint64_t extra) {
        if (!Exl3KvSpillStore::get() || !Exl3KvSpillStore::under_pressure(extra)) return;
        std::vector<std::shared_ptr<Exl3LaneKvSource>> sources;
        {
            std::lock_guard lock(registry_mutex());
            for (const auto& weak : registry())
                if (auto source = weak.lock()) sources.push_back(std::move(source));
        }
        for (const auto& source : sources) {
            source->spill_filled(extra);
            if (!Exl3KvSpillStore::under_pressure(extra)) return;
        }
    }

    // Fills every page of this source holding a row >= `row`.
    void materialize_from(int row) {
        fill_where([&](const Exl3ExactKVPage& page) { return page.first + page.rows > row; });
    }

    // Fills every page of this source except `keep` (a restore rewrites the
    // planes with `keep`'s history, which these planes already hold).
    void materialize_except(const std::vector<std::shared_ptr<const Exl3ExactKVPage>>& keep) {
        std::vector<const Exl3ExactKVPage*> kept;
        kept.reserve(keep.size());
        for (const auto& page : keep) kept.push_back(page.get());
        std::sort(kept.begin(), kept.end());
        fill_where([&](const Exl3ExactKVPage& page) {
            return !std::binary_search(kept.begin(), kept.end(), &page);
        });
    }

    // Hands these planes over to a restored history (`incoming`) without a
    // second host copy of either history: each filled lane-backed page of
    // `incoming` whose rows coincide with a deferred page of this source
    // outside `incoming` trades places with it. The outgoing page receives
    // the plane rows (in the incoming page's buffers), the incoming rows go
    // to the planes and the incoming page is re-homed here. Other deferred
    // pages outside `incoming` are filled first, as materialize_except does.
    void exchange(const std::vector<std::shared_ptr<const Exl3ExactKVPage>>& incoming) {
        std::vector<const Exl3ExactKVPage*> kept;
        kept.reserve(incoming.size());
        for (const auto& page : incoming) kept.push_back(page.get());
        std::sort(kept.begin(), kept.end());
        std::lock_guard lock(mutex_);
        std::vector<std::shared_ptr<Exl3ExactKVPage>> outgoing;
        for (const auto& weak : pages_)
            if (auto page = weak.lock())
                if (holds(*page) && !std::binary_search(kept.begin(), kept.end(), page.get()))
                    outgoing.push_back(std::move(page));
        if (outgoing.empty()) return;
        const auto by_rows = [](const auto& a, const auto& b) {
            return a->first != b->first ? a->first < b->first : a->rows < b->rows;
        };
        std::sort(outgoing.begin(), outgoing.end(), by_rows);
        struct Pair { std::shared_ptr<Exl3ExactKVPage> in, out; };
        std::vector<Pair> pairs;
        std::vector<char> paired(outgoing.size(), 0);
        int exchanged_rows = 0;
        {
            std::shared_lock payload(exl3_page_payload_mutex());
            for (const auto& constant : incoming) {
                if (!constant->source || constant->payload_deferred() || constant->pins.load() > 1 ||
                    !sized(*constant)) continue;
                auto at = std::lower_bound(outgoing.begin(), outgoing.end(), constant, by_rows);
                while (at != outgoing.end() && paired[at - outgoing.begin()]) ++at;
                if (at == outgoing.end() || (*at)->first != constant->first || (*at)->rows != constant->rows)
                    continue;
                paired[at - outgoing.begin()] = 1;
                pairs.push_back({std::const_pointer_cast<Exl3ExactKVPage>(constant), *at});
            }
        }
        std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.in->first < b.in->first; });
        {
            // Unpaired outgoing pages keep their rows before the planes change.
            std::unique_lock payload(exl3_page_payload_mutex());
            std::vector<Exl3ExactKVPage*> fill;
            for (std::size_t index = 0; index < outgoing.size(); ++index)
                if (!paired[index]) fill.push_back(outgoing[index].get());
            fill_locked(fill);
        }
        // Contiguous runs of pairs, at most one stage of rows each.
        std::size_t begin = 0;
        while (begin < pairs.size()) {
            std::size_t end = begin + 1;
            const int lo = pairs[begin].in->first;
            int hi = lo + pairs[begin].in->rows;
            while (end < pairs.size() && pairs[end].in->first == hi &&
                   hi + pairs[end].in->rows - lo <= kStageRows) {
                hi += pairs[end].in->rows;
                ++end;
            }
            std::unique_lock payload(exl3_page_payload_mutex());
            bool valid = true;
            for (std::size_t index = begin; index < end && valid; ++index)
                valid = pairs[index].in->source && !pairs[index].in->payload_deferred() &&
                    pairs[index].in->pins.load() <= 1 && sized(*pairs[index].in) && holds(*pairs[index].out);
            if (valid) {
                // NINFER_EXL3_TEST_LANE_EXCHANGE=1 checks every run against copies.
                std::vector<std::uint16_t> before_planes, before_pages;
                if (exchange_test()) {
                    before_planes = read_rows(lo, hi);
                    before_pages = gather(pairs.data() + begin, pairs.data() + end, lo, hi);
                }
                exchange_run_locked(pairs.data() + begin, pairs.data() + end, lo, hi);
                if (exchange_test() && (read_rows(lo, hi) != before_pages ||
                    gather(pairs.data() + begin, pairs.data() + end, lo, hi) != before_planes))
                    throw std::logic_error("lane KV exchange test mismatch");
                exchanged_rows += hi - lo;
                for (std::size_t index = begin; index < end; ++index) {
                    auto& in = *pairs[index].in;
                    auto& out = *pairs[index].out;
                    for (int bank = 0; bank < planes_.banks; ++bank) {
                        out.k[bank].swap(in.k[bank]);
                        out.v[bank].swap(in.v[bank]);
                    }
                    out.deferred.store(false);
                    release_payload(in);
                    in.source = shared_from_this();
                    in.source_id.store(this);
                    in.deferred.store(true);
                    track_locked(pairs[index].in);
                }
            } else {
                // The planes stay unchanged for these rows: keep the outgoing rows.
                std::vector<Exl3ExactKVPage*> fill;
                for (std::size_t index = begin; index < end; ++index)
                    if (holds(*pairs[index].out)) fill.push_back(pairs[index].out.get());
                fill_locked(fill);
            }
            begin = end;
        }
        if (exchange_test())
            std::fprintf(stderr, "LANE_KV_EXCHANGE pairs=%zu rows=%d filled=%zu\n", pairs.size(), exchanged_rows,
                outgoing.size() - pairs.size());
    }

    // Some page of this source holding a row >= `row` is still deferred.
    bool deferred_beyond(int row) {
        std::lock_guard lock(mutex_);
        for (const auto& weak : pages_)
            if (auto page = weak.lock())
                if (holds(*page) && page->first + page->rows > row) return true;
        return false;
    }

    // Copies the rows of a contiguous run of this source's deferred pages
    // straight into another lane's planes (a restore there), without filling
    // them: from these planes, or from the spill file for spilled pages. False
    // if a page left this source meanwhile (use its values).
    bool copy_run_to(const std::vector<const Exl3ExactKVPage*>& run, const Planes& destination,
                     cudaStream_t stream) {
        std::lock_guard lock(mutex_);   // no fill or re-home of these pages meanwhile
        std::shared_lock payload(exl3_page_payload_mutex());
        bool spilled = false;
        for (const auto* page : run) {
            if (spilled_here(*page)) spilled = true;
            else if (!holds(*page)) return false;
        }
        if (run.empty()) return true;
        if (spilled) {
            par_each(run.begin(), run.end(), [&](const Exl3ExactKVPage* page) {
                if (spilled_here(*page)) read_slot_to_planes(*page, destination);
                else {
                    copy_rows_to_planes(*page, destination, cudaStreamPerThread);
                    check(cudaStreamSynchronize(cudaStreamPerThread), "lane KV plane copy");
                }
            });
            return true;
        }
        const int lo = run.front()->first, hi = run.back()->first + run.back()->rows;
        const std::size_t offset = static_cast<std::size_t>(lo) * kRowWords;
        const std::size_t bytes = static_cast<std::size_t>(hi - lo) * kRowWords * sizeof(std::uint16_t);
        for (int bank = 0; bank < planes_.banks; ++bank) {
            check(cudaMemcpyAsync(const_cast<std::uint16_t*>(destination.k[bank]) + offset, planes_.k[bank] + offset,
                bytes, cudaMemcpyDefault, stream), "lane KV plane copy");
            check(cudaMemcpyAsync(const_cast<std::uint16_t*>(destination.v[bank]) + offset, planes_.v[bank] + offset,
                bytes, cudaMemcpyDefault, stream), "lane KV plane copy");
        }
        check(cudaStreamSynchronize(stream), "lane KV plane copy");
        return true;
    }

private:
    // std::for_each(par) that rethrows the first exception of any element.
    template <class Iterator, class Function> static void par_each(Iterator first, Iterator last, Function function) {
        std::exception_ptr error;
        std::mutex error_mutex;
        std::for_each(std::execution::par, first, last, [&](auto&& item) {
            try {
                function(item);
            } catch (...) {
                std::lock_guard lock(error_mutex);
                if (!error) error = std::current_exception();
            }
        });
        if (error) std::rethrow_exception(error);
    }
    static std::mutex& registry_mutex() {
        static std::mutex mutex;
        return mutex;
    }
    static std::vector<std::weak_ptr<Exl3LaneKvSource>>& registry() {
        static std::vector<std::weak_ptr<Exl3LaneKvSource>> sources;
        return sources;
    }

    std::size_t page_bytes(const Exl3ExactKVPage& page) const {
        return static_cast<std::size_t>(page.rows) * kRowWords * sizeof(std::uint16_t) * 2 * planes_.banks;
    }
    // Slot layout: [bank][K, V][rows][1024] FP16.
    std::size_t slot_offset(const Exl3ExactKVPage& page, int bank, int plane) const {
        return static_cast<std::size_t>(bank * 2 + plane) * page.rows * kRowWords;
    }

    // One page's rows of these planes into another planes set (thread stream).
    void copy_rows_to_planes(const Exl3ExactKVPage& page, const Planes& destination, cudaStream_t stream) const {
        const std::size_t offset = static_cast<std::size_t>(page.first) * kRowWords;
        const std::size_t bytes = static_cast<std::size_t>(page.rows) * kRowWords * sizeof(std::uint16_t);
        for (int bank = 0; bank < planes_.banks; ++bank) {
            check(cudaMemcpyAsync(const_cast<std::uint16_t*>(destination.k[bank]) + offset, planes_.k[bank] + offset,
                bytes, cudaMemcpyDefault, stream), "lane KV plane copy");
            check(cudaMemcpyAsync(const_cast<std::uint16_t*>(destination.v[bank]) + offset, planes_.v[bank] + offset,
                bytes, cudaMemcpyDefault, stream), "lane KV plane copy");
        }
    }
    // A spilled page's slot straight into planes (restore), on this thread.
    void read_slot_to_planes(const Exl3ExactKVPage& page, const Planes& destination) const {
        auto* buffer = Exl3KvSpillStore::aligned_buffer();
        Exl3KvSpillStore::get()->read(page.spill.load(), buffer, page_bytes(page));
        const auto* words = reinterpret_cast<const std::uint16_t*>(buffer);
        const std::size_t offset = static_cast<std::size_t>(page.first) * kRowWords;
        const std::size_t bytes = static_cast<std::size_t>(page.rows) * kRowWords * sizeof(std::uint16_t);
        for (int bank = 0; bank < destination.banks; ++bank)
            for (int plane = 0; plane < 2; ++plane) {
                auto* rows = const_cast<std::uint16_t*>(plane ? destination.v[bank] : destination.k[bank]) + offset;
                if (destination.cpu_visible) std::memcpy(rows, words + slot_offset(page, bank, plane), bytes);
                else check(cudaMemcpyAsync(rows, words + slot_offset(page, bank, plane), bytes, cudaMemcpyDefault,
                    cudaStreamPerThread), "lane KV spill restore");
            }
        // The buffer is reused by this thread's next page.
        if (!destination.cpu_visible) check(cudaStreamSynchronize(cudaStreamPerThread), "lane KV spill restore");
    }

    // Caller holds mutex_ and the payload mutex exclusively.
    void unspill_locked(Exl3ExactKVPage& page) {
        auto* buffer = Exl3KvSpillStore::aligned_buffer();
        Exl3KvSpillStore::get()->read(page.spill.load(), buffer, page_bytes(page));
        const auto* words = reinterpret_cast<const std::uint16_t*>(buffer);
        const std::size_t count = static_cast<std::size_t>(page.rows) * kRowWords;
        for (int bank = 0; bank < planes_.banks; ++bank) {
            page.k[bank].assign(words + slot_offset(page, bank, 0), words + slot_offset(page, bank, 0) + count);
            page.v[bank].assign(words + slot_offset(page, bank, 1), words + slot_offset(page, bank, 1) + count);
        }
        page.spill.reset();
        page.deferred.store(false);
    }

    // Spills this source's filled pages until `extra` bytes fit above the floor.
    void spill_filled(std::uint64_t extra) {
        std::lock_guard lock(mutex_);
        std::vector<std::shared_ptr<Exl3ExactKVPage>> filled;
        for (const auto& weak : pages_)
            if (auto page = weak.lock())
                if (page->source_id.load() == this && !page->payload_deferred()) filled.push_back(std::move(page));
        constexpr std::size_t kBatch = 32;
        for (std::size_t begin = 0; begin < filled.size(); begin += kBatch) {
            if (!Exl3KvSpillStore::under_pressure(extra)) return;
            std::unique_lock payload(exl3_page_payload_mutex());
            const std::size_t end = std::min(filled.size(), begin + kBatch);
            par_each(filled.begin() + begin, filled.begin() + end, [&](const auto& page) {
                if (page->source_id.load() != this || page->payload_deferred() || page->pins.load() ||
                    !sized(*page)) return;
                auto* buffer = Exl3KvSpillStore::aligned_buffer();
                auto* words = reinterpret_cast<std::uint16_t*>(buffer);
                const std::size_t count = static_cast<std::size_t>(page->rows) * kRowWords;
                for (int bank = 0; bank < planes_.banks; ++bank) {
                    std::memcpy(words + slot_offset(*page, bank, 0), page->k[bank].data(), count * sizeof(std::uint16_t));
                    std::memcpy(words + slot_offset(*page, bank, 1), page->v[bank].data(), count * sizeof(std::uint16_t));
                }
                auto* store = Exl3KvSpillStore::get();
                const auto slot = store->allocate();
                try {
                    store->write(slot, buffer, page_bytes(*page));
                } catch (...) {
                    store->release(slot);
                    return;
                }
                release_payload(*page);
                page->spill.store(slot);
                page->deferred.store(true);
            });
        }
    }

    bool sized(const Exl3ExactKVPage& page) const {
        const std::size_t words = static_cast<std::size_t>(page.rows) * kRowWords;
        for (int bank = 0; bank < planes_.banks; ++bank)
            if (page.k[bank].size() != words || page.v[bank].size() != words) return false;
        return true;
    }

    // Caller holds mutex_ and the payload mutex exclusively. Swaps plane rows
    // [lo, hi) with the buffers of a run's (contiguous) incoming pages: the
    // planes then hold the incoming rows and the buffers the previous ones.
    template <class Pair> void exchange_run_locked(Pair* begin, Pair* end, int lo, int hi) {
        const std::size_t run_bytes = static_cast<std::size_t>(hi - lo) * kRowWords * sizeof(std::uint16_t);
        if (planes_.cpu_visible) {
            std::for_each(std::execution::par, begin, end, [&](Pair& pair) {
                auto& in = *pair.in;
                const std::size_t offset = static_cast<std::size_t>(in.first) * kRowWords;
                const std::size_t words = static_cast<std::size_t>(in.rows) * kRowWords;
                std::vector<std::uint16_t> old(words);
                for (int bank = 0; bank < planes_.banks; ++bank)
                    for (int plane = 0; plane < 2; ++plane) {
                        auto* rows = const_cast<std::uint16_t*>(plane ? planes_.v[bank] : planes_.k[bank]) + offset;
                        auto& buffer = plane ? in.v[bank] : in.k[bank];
                        std::memcpy(old.data(), rows, words * sizeof(std::uint16_t));
                        std::memcpy(rows, buffer.data(), words * sizeof(std::uint16_t));
                        std::memcpy(buffer.data(), old.data(), words * sizeof(std::uint16_t));
                    }
            });
            return;
        }
        ensure_staging(true);
        for (int bank = 0; bank < planes_.banks; ++bank)
            for (int plane = 0; plane < 2; ++plane) {
                auto* rows = const_cast<std::uint16_t*>(plane ? planes_.v[bank] : planes_.k[bank]) +
                    static_cast<std::size_t>(lo) * kRowWords;
                check(cudaMemcpyAsync(stage_, rows, run_bytes, cudaMemcpyDefault, stream_), "lane KV exchange");
                std::for_each(std::execution::par, begin, end, [&](Pair& pair) {
                    const auto& buffer = plane ? pair.in->v[bank] : pair.in->k[bank];
                    std::memcpy(swap_stage_ + static_cast<std::size_t>(pair.in->first - lo) * kRowWords,
                        buffer.data(), buffer.size() * sizeof(std::uint16_t));
                });
                check(cudaStreamSynchronize(stream_), "lane KV exchange");
                check(cudaMemcpyAsync(rows, swap_stage_, run_bytes, cudaMemcpyDefault, stream_), "lane KV exchange");
                std::for_each(std::execution::par, begin, end, [&](Pair& pair) {
                    auto& buffer = plane ? pair.in->v[bank] : pair.in->k[bank];
                    std::memcpy(buffer.data(), stage_ + static_cast<std::size_t>(pair.in->first - lo) * kRowWords,
                        buffer.size() * sizeof(std::uint16_t));
                });
                check(cudaStreamSynchronize(stream_), "lane KV exchange");
            }
    }

    static bool exchange_test() {
        static const bool enabled = [] {
            const char* value = std::getenv("NINFER_EXL3_TEST_LANE_EXCHANGE");
            return value && std::strcmp(value, "1") == 0;
        }();
        return enabled;
    }
    // Test helpers: plane rows [lo, hi) and a run's page buffers, bank-major.
    std::vector<std::uint16_t> read_rows(int lo, int hi) {
        const std::size_t words = static_cast<std::size_t>(hi - lo) * kRowWords;
        std::vector<std::uint16_t> out(words * planes_.banks * 2);
        for (int bank = 0; bank < planes_.banks; ++bank)
            for (int plane = 0; plane < 2; ++plane)
                check(cudaMemcpy(out.data() + (bank * 2 + plane) * words,
                    (plane ? planes_.v[bank] : planes_.k[bank]) + static_cast<std::size_t>(lo) * kRowWords,
                    words * sizeof(std::uint16_t), cudaMemcpyDefault), "lane KV exchange test");
        return out;
    }
    template <class Pair> std::vector<std::uint16_t> gather(Pair* begin, Pair* end, int lo, int hi) {
        const std::size_t words = static_cast<std::size_t>(hi - lo) * kRowWords;
        std::vector<std::uint16_t> out(words * planes_.banks * 2);
        for (auto* pair = begin; pair != end; ++pair)
            for (int bank = 0; bank < planes_.banks; ++bank)
                for (int plane = 0; plane < 2; ++plane) {
                    const auto& buffer = plane ? pair->in->v[bank] : pair->in->k[bank];
                    std::copy(buffer.begin(), buffer.end(), out.begin() + (bank * 2 + plane) * words +
                        static_cast<std::size_t>(pair->in->first - lo) * kRowWords);
                }
        return out;
    }

    void ensure_staging(bool swap) {
        if (!stream_) check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "lane KV source stream");
        const std::size_t bytes = static_cast<std::size_t>(kStageRows) * kRowWords * sizeof(std::uint16_t);
        if (!stage_) check(cudaMallocHost(reinterpret_cast<void**>(&stage_), bytes), "lane KV source stage");
        if (swap && !swap_stage_)
            check(cudaMallocHost(reinterpret_cast<void**>(&swap_stage_), bytes), "lane KV source stage");
    }

    static void release_payload(Exl3ExactKVPage& page) {
        for (int bank = 0; bank < 16; ++bank) {
            std::vector<std::uint16_t>().swap(page.k[bank]);
            std::vector<std::uint16_t>().swap(page.v[bank]);
        }
    }
    void track_locked(const std::shared_ptr<Exl3ExactKVPage>& page) {
        if (pages_.size() >= 2 * live_hint_ + 256) prune_locked();
        pages_.push_back(page);
    }
    void prune_locked() {
        pages_.erase(std::remove_if(pages_.begin(), pages_.end(), [&](const auto& weak) {
            const auto page = weak.lock();
            return !page || page->source_id.load() != this;
        }), pages_.end());
        live_hint_ = pages_.size();
    }

    template <class Predicate> void fill_where(Predicate predicate) {
        std::lock_guard lock(mutex_);
        std::unique_lock payload(exl3_page_payload_mutex());
        std::vector<std::shared_ptr<Exl3ExactKVPage>> owners;
        std::vector<Exl3ExactKVPage*> pages;
        for (const auto& weak : pages_)
            if (auto page = weak.lock())
                if (holds(*page) && predicate(*page)) {
                    pages.push_back(page.get());
                    owners.push_back(std::move(page));
                }
        fill_locked(pages);
        prune_locked();
    }

    static void check(cudaError_t error, const char* what) {
        if (error != cudaSuccess)
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }

    // Caller holds mutex_ and the payload mutex exclusively: the pages' rows
    // go from these planes to spill slots, in windows of kSpillPages pages.
    static constexpr std::size_t kSpillPages = 16;
    void spill_from_planes_locked(const std::vector<Exl3ExactKVPage*>& pages) {
        auto* store = Exl3KvSpillStore::get();
        // Pool threads below see their own thread_locals: bind this thread's.
        thread_local std::vector<std::uint8_t*> window_buffers;
        auto& buffers = window_buffers;
        const auto window_buffer = [&](std::size_t index) {
            while (buffers.size() <= index) {
                void* data = VirtualAlloc(nullptr, Exl3KvSpillStore::kSlotBytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
                if (!data) throw std::runtime_error("KV spill window buffer");
                buffers.push_back(static_cast<std::uint8_t*>(data));
            }
            return buffers[index];
        };
        for (std::size_t begin = 0; begin < pages.size(); begin += kSpillPages) {
            const std::size_t end = std::min(pages.size(), begin + kSpillPages);
            for (std::size_t index = begin; index < end; ++index) window_buffer(index - begin);
            // Rows of the window's pages (one DMA per bank and plane when the
            // window is contiguous), packed into the slot buffers.
            int lo = pages[begin]->first, hi = lo, covered = 0;
            for (std::size_t index = begin; index < end; ++index) {
                hi = std::max(hi, pages[index]->first + pages[index]->rows);
                covered += pages[index]->rows;
            }
            const bool contiguous = hi - lo == covered;
            if (!planes_.cpu_visible) ensure_staging(false);
            for (int bank = 0; bank < planes_.banks; ++bank)
                for (int plane = 0; plane < 2; ++plane) {
                    const auto* source_rows = plane ? planes_.v[bank] : planes_.k[bank];
                    if (!planes_.cpu_visible && contiguous) {
                        check(cudaMemcpyAsync(stage_, source_rows + static_cast<std::size_t>(lo) * kRowWords,
                            static_cast<std::size_t>(hi - lo) * kRowWords * sizeof(std::uint16_t), cudaMemcpyDefault,
                            stream_), "lane KV spill");
                        check(cudaStreamSynchronize(stream_), "lane KV spill");
                    }
                    for (std::size_t index = begin; index < end; ++index) {
                        const auto& page = *pages[index];
                        auto* words = reinterpret_cast<std::uint16_t*>(buffers[index - begin]) +
                            slot_offset(page, bank, plane);
                        const std::size_t bytes = static_cast<std::size_t>(page.rows) * kRowWords * sizeof(std::uint16_t);
                        const auto* rows = source_rows + static_cast<std::size_t>(page.first) * kRowWords;
                        if (planes_.cpu_visible) std::memcpy(words, rows, bytes);
                        else if (contiguous)
                            std::memcpy(words, stage_ + static_cast<std::size_t>(page.first - lo) * kRowWords, bytes);
                        else {
                            check(cudaMemcpyAsync(stage_, rows, bytes, cudaMemcpyDefault, stream_), "lane KV spill");
                            check(cudaStreamSynchronize(stream_), "lane KV spill");
                            std::memcpy(words, stage_, bytes);
                        }
                    }
                }
            std::vector<std::int64_t> slots(end - begin);
            for (auto& slot : slots) slot = store->allocate();
            std::vector<std::size_t> order(end - begin);
            for (std::size_t index = 0; index < order.size(); ++index) order[index] = index;
            par_each(order.begin(), order.end(), [&](std::size_t index) {
                store->write(slots[index], buffers[index], page_bytes(*pages[begin + index]));
            });
            for (std::size_t index = begin; index < end; ++index) {
                release_payload(*pages[index]);
                pages[index]->spill.store(slots[index - begin]);
            }
        }
    }

    // Caller holds mutex_ and the payload mutex exclusively. Under host
    // memory pressure (and `may_spill`) the rows go to the spill file.
    void fill_locked(std::vector<Exl3ExactKVPage*>& pages, bool may_spill = true) {
        if (pages.empty()) return;
        std::sort(pages.begin(), pages.end(), [](const auto* a, const auto* b) { return a->first < b->first; });
        if (may_spill && Exl3KvSpillStore::get()) {
            std::uint64_t bytes = 0;
            for (const auto* page : pages) bytes += page_bytes(*page);
            if (Exl3KvSpillStore::under_pressure(bytes)) {
                spill_from_planes_locked(pages);
                return;
            }
        }
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
            ensure_staging(false);
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
    std::uint16_t* swap_stage_ = nullptr;
};

// Pins pages for the scope of a reader (see Exl3PinCount).
class Exl3PagePins {
public:
    explicit Exl3PagePins(const std::vector<std::shared_ptr<const Exl3ExactKVPage>>& pages) : pages_(pages) {
        for (const auto& page : pages_) page->pins.value.fetch_add(1, std::memory_order_acq_rel);
    }
    ~Exl3PagePins() {
        for (const auto& page : pages_) page->pins.value.fetch_sub(1, std::memory_order_acq_rel);
    }
    Exl3PagePins(const Exl3PagePins&) = delete;
    Exl3PagePins& operator=(const Exl3PagePins&) = delete;

private:
    std::vector<std::shared_ptr<const Exl3ExactKVPage>> pages_;
};

// Fills a lane-backed page before its values are read (whichever lane holds it).
inline void exl3_require_kv_payload(const Exl3ExactKVPage& page) {
    // A re-home between reading the owner and filling retries with the new owner.
    while (page.payload_deferred()) {
        std::shared_ptr<Exl3LaneKvSource> owner;
        {
            std::shared_lock payload(exl3_page_payload_mutex());
            owner = page.source;
        }
        if (!owner) return;
        owner->materialize(page);
    }
}

}  // namespace ninfer::exl3
