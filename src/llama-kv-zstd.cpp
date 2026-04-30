#ifdef GGML_USE_ZSTD

#include "llama-kv-zstd.h"
#include "llama-impl.h"

#include <zstd.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

static void kv_zstd_release_raw_pages(uint8_t * ptr, size_t size) {
    static const size_t page_size = (size_t) sysconf(_SC_PAGESIZE);
    static std::atomic<bool> logged_madvise_error{false};

    if (ptr == nullptr || size == 0 || page_size == 0) {
        return;
    }

    const uintptr_t begin = (uintptr_t) ptr;
    const uintptr_t end   = begin + size;

    const uintptr_t page_mask     = ~(uintptr_t)(page_size - 1);
    const uintptr_t aligned_begin = (begin + page_size - 1) & page_mask;
    const uintptr_t aligned_end   = end & page_mask;

    if (aligned_end <= aligned_begin) {
        return;
    }

    errno = 0;
    if (madvise((void *) aligned_begin, aligned_end - aligned_begin, MADV_DONTNEED) != 0 &&
            !logged_madvise_error.exchange(true, std::memory_order_relaxed)) {
        LLAMA_LOG_WARN("%s: madvise(MADV_DONTNEED) failed: %s\n", __func__, strerror(errno));
    }
}

kv_zstd_tensor::kv_zstd_tensor(
        uint8_t * raw_, size_t raw_bytes_, size_t frame_bytes_, uint32_t n_ctx_, uint32_t n_stream_,
        uint32_t n_embd_, size_t bytes_per_slot_, size_t bytes_per_el_, kv_zstd_tensor_layout layout_)
    : raw(raw_), raw_bytes(raw_bytes_), frame_bytes(frame_bytes_), n_ctx(n_ctx_), n_stream(n_stream_),
      n_embd(n_embd_), bytes_per_slot(bytes_per_slot_), bytes_per_el(bytes_per_el_), layout(layout_) {
    n_frames = (int32_t)((raw_bytes + frame_bytes - 1) / frame_bytes);
    bytes_per_stream = n_stream > 0 ? raw_bytes / n_stream : raw_bytes;
    cframes.resize(n_frames);
}

void kv_zstd_tensor::select_frames(const std::vector<kv_zstd_cell_range> & ranges) {
    active_frames.clear();
    covered_bytes     = 0;
    actual_used_bytes = 0;

    if (raw == nullptr || raw_bytes == 0 || frame_bytes == 0 || n_frames <= 0 || n_ctx == 0) {
        return;
    }

    std::vector<uint8_t> active((size_t)n_frames, 0);

    auto mark_bytes = [&](size_t off, size_t len) {
        if (len == 0 || off >= raw_bytes) {
            return;
        }

        len = std::min(len, raw_bytes - off);
        actual_used_bytes += len;

        const size_t first = off / frame_bytes;
        const size_t last  = (off + len - 1) / frame_bytes;

        for (size_t i = first; i <= last && i < active.size(); ++i) {
            active[i] = 1;
        }
    };

    for (const kv_zstd_cell_range & r : ranges) {
        if (r.begin >= r.end || r.begin >= n_ctx || r.stream >= n_stream) {
            continue;
        }

        const uint32_t begin = r.begin;
        const uint32_t end   = std::min(r.end, n_ctx);
        if (begin >= end) {
            continue;
        }

        if (layout == kv_zstd_tensor_layout::SLOT_MAJOR) {
            const size_t off = (size_t)r.stream * bytes_per_stream + (size_t)begin * bytes_per_slot;
            const size_t len = (size_t)(end - begin) * bytes_per_slot;
            mark_bytes(off, len);
        } else {
            for (uint32_t j = 0; j < n_embd; ++j) {
                const size_t off = ((size_t)r.stream * n_ctx * n_embd + (size_t)j * n_ctx + begin) * bytes_per_el;
                const size_t len = (size_t)(end - begin) * bytes_per_el;
                mark_bytes(off, len);
            }
        }
    }

    for (int32_t i = 0; i < n_frames; ++i) {
        if (active[(size_t)i]) {
            active_frames.push_back(i);
            const size_t off = (size_t)i * frame_bytes;
            covered_bytes += std::min(frame_bytes, raw_bytes - off);
        }
    }
}

void kv_zstd_state::bg_loop() {
    while (true) {
        std::vector<kv_zstd_cell_range> ranges;
        uint32_t current_n_used = 0;
        {
            std::unique_lock<std::mutex> lk(mu);
            bg_idle = true;
            idle_cv.notify_all();
            wake_cv.wait(lk, [this]{ return work_ready || bg_stop; });
            if (bg_stop) {
                return;
            }
            ranges         = work_ranges;
            current_n_used = n_used;
            work_ready     = false;
            bg_idle        = false;
        }

        for (auto & t : tensors) {
            t.select_frames(ranges);

            int32_t done  = t.n_done.load(std::memory_order_relaxed);
            int32_t total = (int32_t)t.active_frames.size();

            while (done < total) {
                if (interrupted.load(std::memory_order_acquire)) {
                    break;
                }

                const int32_t i = t.active_frames[(size_t)done];
                const size_t off = (size_t)i * t.frame_bytes;
                const size_t sz  = std::min(t.frame_bytes, t.raw_bytes - off);

                const size_t bound = ZSTD_compressBound(sz);
                t.cframes[i].resize(bound);
                const size_t csize = ZSTD_compress(t.cframes[i].data(), bound, t.raw + off, sz, level);
                if (ZSTD_isError(csize)) {
                    LLAMA_LOG_ERROR("%s: ZSTD_compress failed: %s\n", __func__, ZSTD_getErrorName(csize));
                    t.cframes[i].clear();
                    return;
                }

                if ((float)csize < threshold * (float)sz) {
                    t.cframes[i].resize(csize);
                    t.cframes[i].shrink_to_fit();
                    kv_zstd_release_raw_pages(t.raw + off, sz);
                } else {
                    t.cframes[i].clear();
                    t.cframes[i].shrink_to_fit();
                }

                t.n_done.store(done + 1, std::memory_order_release);
                done++;
            }

            if (interrupted.load(std::memory_order_acquire)) {
                break;
            }
        }

        if (!interrupted.load(std::memory_order_acquire)) {
            const size_t used    = raw_used_bytes();
            const size_t covered = covered_bytes();
            const size_t comp    = compressed_bytes();
            if (covered > 0) {
                size_t frames = 0;
                for (const auto & t : tensors) {
                    frames += t.active_frames.size();
                }
                const double saved = covered > comp ? (double)(covered - comp) : 0.0;
                LLAMA_LOG_INFO(
                    "kv-zstd: compressed %.1f MiB -> %.1f MiB (%.1f%%) "
                    "[used=%.1f MiB covered=%.1f MiB saved=%.1f MiB/%.1f%% slots=%u ranges=%zu frames=%zu]\n",
                    covered / (1024.0 * 1024.0),
                    comp    / (1024.0 * 1024.0),
                    covered > 0 ? 100.0 * comp / covered : 0.0,
                    used    / (1024.0 * 1024.0),
                    covered / (1024.0 * 1024.0),
                    saved   / (1024.0 * 1024.0),
                    covered > 0 ? 100.0 * saved / covered : 0.0,
                    (unsigned)current_n_used,
                    ranges.size(),
                    frames);
            }
        }
    }
}

void kv_zstd_state::init(int zstd_level, float thresh) {
    level     = zstd_level;
    threshold = thresh;
    bg_thread = std::thread([this]{ bg_loop(); });
}

kv_zstd_state::~kv_zstd_state() {
    {
        std::lock_guard<std::mutex> lk(mu);
        bg_stop    = true;
        work_ready = false;
    }
    wake_cv.notify_one();
    if (bg_thread.joinable()) {
        bg_thread.join();
    }
}

void kv_zstd_state::sync() {
    interrupted.store(true, std::memory_order_release);

    {
        std::unique_lock<std::mutex> lk(mu);
        idle_cv.wait(lk, [this]{ return bg_idle; });
    }

    size_t bytes_restored = 0;
    for (auto & t : tensors) {
        for (int32_t i = 0; i < t.n_frames; i++) {
            if (!t.cframes[i].empty()) {
                const size_t off = (size_t)i * t.frame_bytes;
                const size_t sz  = std::min(t.frame_bytes, t.raw_bytes - off);
                const size_t ret = ZSTD_decompress(t.raw + off, sz, t.cframes[i].data(), t.cframes[i].size());
                if (ZSTD_isError(ret)) {
                    LLAMA_LOG_ERROR("%s: ZSTD_decompress failed: %s\n", __func__, ZSTD_getErrorName(ret));
                }
                t.cframes[i].clear();
                t.cframes[i].shrink_to_fit();
                bytes_restored += sz;
            }
        }
        t.n_done.store(0, std::memory_order_release);
    }

    if (bytes_restored > 0) {
        LLAMA_LOG_DEBUG("%s: restored %.1f MiB from kv zstd cache\n",
            __func__, bytes_restored / (1024.0 * 1024.0));
    }

    interrupted.store(false, std::memory_order_release);
}

size_t kv_zstd_state::compressed_bytes() const {
    size_t total = 0;
    for (const auto & t : tensors) {
        for (const auto & cf : t.cframes) {
            total += cf.size();
        }
    }
    return total;
}

size_t kv_zstd_state::covered_bytes() const {
    size_t total = 0;
    for (const auto & t : tensors) {
        total += t.covered_bytes;
    }
    return total;
}

size_t kv_zstd_state::raw_used_bytes() const {
    size_t total = 0;
    for (const auto & t : tensors) {
        total += t.actual_used_bytes;
    }
    return total;
}

void kv_zstd_state::start(std::vector<kv_zstd_cell_range> ranges, uint32_t nu) {
    {
        std::lock_guard<std::mutex> lk(mu);
        n_used      = nu;
        work_ranges = std::move(ranges);
        work_ready  = true;
    }
    wake_cv.notify_one();
}

#endif // GGML_USE_ZSTD
