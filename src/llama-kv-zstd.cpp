#ifdef GGML_USE_ZSTD

#include "llama-kv-zstd.h"
#include "llama-impl.h"

#include <zstd.h>
#include <sys/mman.h>

#include <algorithm>
#include <cassert>

// ---------------------------------------------------------------------------
// kv_zstd_tensor
// ---------------------------------------------------------------------------

kv_zstd_tensor::kv_zstd_tensor(uint8_t * raw_, size_t raw_bytes_, size_t frame_bytes_)
    : raw(raw_), raw_bytes(raw_bytes_), frame_bytes(frame_bytes_) {
    n_frames = (int32_t)((raw_bytes + frame_bytes - 1) / frame_bytes);
    cframes.resize(n_frames);
}

// ---------------------------------------------------------------------------
// kv_zstd_state — background thread
// ---------------------------------------------------------------------------

void kv_zstd_state::bg_loop() {
    while (true) {
        {
            std::unique_lock<std::mutex> lk(mu);
            bg_idle = true;
            idle_cv.notify_all();  // wake any sync() waiting for idle
            wake_cv.wait(lk, [this]{ return work_ready || bg_stop; });
            if (bg_stop) { return; }
            work_ready = false;
            bg_idle    = false;
        }

        auto compress_all = [&]() {
            const int32_t total_passes = (recompress_level > 0) ? 2 : 1;

            for (auto & t : tensors) {
                int32_t done  = t.n_done.load(std::memory_order_relaxed);
                int32_t total = t.n_frames * total_passes;

                while (done < total) {
                    if (interrupted.load(std::memory_order_acquire)) { return; }

                    int32_t i           = done % t.n_frames;
                    bool    second_pass = (done >= t.n_frames);
                    size_t  off         = (size_t)i * t.frame_bytes;
                    size_t  sz          = std::min(t.frame_bytes, t.raw_bytes - off);
                    int     cur_level   = second_pass ? recompress_level : level;

                    if (!second_pass) {
                        // ---- first pass: compress raw at `level` ----
                        size_t bound = ZSTD_compressBound(sz);
                        t.cframes[i].resize(bound);
                        size_t csize = ZSTD_compress(
                            t.cframes[i].data(), bound, t.raw + off, sz, cur_level);
                        if (ZSTD_isError(csize)) {
                            LLAMA_LOG_ERROR("%s: ZSTD_compress failed: %s\n",
                                __func__, ZSTD_getErrorName(csize));
                            t.cframes[i].clear();
                            return;
                        }
                        if ((float)csize < threshold * (float)sz) {
                            t.cframes[i].resize(csize);
                            madvise(t.raw + off, sz, MADV_DONTNEED);
                        } else {
                            t.cframes[i].clear(); // not worth it; keep raw valid
                        }

                    } else {
                        // ---- second pass: (re)compress at `recompress_level` ----
                        bool had_first = !t.cframes[i].empty();

                        if (had_first) {
                            // Decompress first-pass result into raw (used as scratch).
                            size_t ret = ZSTD_decompress(t.raw + off, sz,
                                                          t.cframes[i].data(), t.cframes[i].size());
                            if (ZSTD_isError(ret)) {
                                // Keep first-pass result intact; re-release the scratch pages.
                                madvise(t.raw + off, sz, MADV_DONTNEED);
                                t.n_done.store(done + 1, std::memory_order_release);
                                done++;
                                continue;
                            }
                        }
                        // raw is valid now (either always was, or just restored).

                        size_t bound = ZSTD_compressBound(sz);
                        std::vector<uint8_t> new_cf(bound);
                        size_t csize = ZSTD_compress(
                            new_cf.data(), bound, t.raw + off, sz, cur_level);

                        bool worth_it = !ZSTD_isError(csize) &&
                                        (float)csize < threshold * (float)sz;
                        bool better   = worth_it &&
                                        (!had_first || csize < t.cframes[i].size());

                        if (better) {
                            new_cf.resize(csize);
                            t.cframes[i] = std::move(new_cf);
                        }
                        // Release raw pages if this frame is now compressed by either pass.
                        if (!t.cframes[i].empty()) {
                            madvise(t.raw + off, sz, MADV_DONTNEED);
                        }
                    }

                    t.n_done.store(done + 1, std::memory_order_release);
                    done++;
                }
            }
        };
        compress_all();
    }
}

// ---------------------------------------------------------------------------
// kv_zstd_state — public API
// ---------------------------------------------------------------------------

void kv_zstd_state::init(int zstd_level, float thresh, int recompress) {
    level            = zstd_level;
    threshold        = thresh;
    recompress_level = recompress;
    bg_thread        = std::thread([this]{ bg_loop(); });
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
    // Ask the bg thread to stop at the next frame boundary.
    interrupted.store(true, std::memory_order_release);

    // Wait for it to go idle.
    {
        std::unique_lock<std::mutex> lk(mu);
        idle_cv.wait(lk, [this]{ return bg_idle; });
    }

    // Restore any frames that were compressed + MADV_DONTNEED'd.
    // Cap at n_frames: n_done may exceed n_frames when the second pass is running,
    // but cframes has exactly n_frames entries.
    size_t bytes_restored = 0;
    for (auto & t : tensors) {
        int32_t done        = t.n_done.load(std::memory_order_acquire);
        int32_t n_compressed = std::min(done, t.n_frames);
        for (int32_t i = 0; i < n_compressed; i++) {
            if (!t.cframes[i].empty()) {
                size_t off = (size_t)i * t.frame_bytes;
                size_t sz  = std::min(t.frame_bytes, t.raw_bytes - off);
                size_t ret = ZSTD_decompress(t.raw + off, sz,
                                              t.cframes[i].data(), t.cframes[i].size());
                if (ZSTD_isError(ret)) {
                    LLAMA_LOG_ERROR("%s: ZSTD_decompress failed: %s\n",
                        __func__, ZSTD_getErrorName(ret));
                }
                t.cframes[i].clear();
                t.cframes[i].shrink_to_fit();
                bytes_restored += sz;
            }
            // else: frame was below threshold, raw was never released
        }
        t.n_done.store(0, std::memory_order_release);
    }

    if (bytes_restored > 0) {
        LLAMA_LOG_DEBUG("%s: restored %.1f MB from kv zstd cache\n",
            __func__, bytes_restored / (1024.0 * 1024.0));
    }

    interrupted.store(false, std::memory_order_release);
}

void kv_zstd_state::start() {
    {
        std::lock_guard<std::mutex> lk(mu);
        work_ready = true;
    }
    wake_cv.notify_one();
}

#endif // GGML_USE_ZSTD
