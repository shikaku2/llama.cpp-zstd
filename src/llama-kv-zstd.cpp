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
            for (auto & t : tensors) {
                int32_t done = t.n_done.load(std::memory_order_relaxed);
                while (done < t.n_frames) {
                    if (interrupted.load(std::memory_order_acquire)) { return; }

                    size_t off = (size_t)done * t.frame_bytes;
                    size_t sz  = std::min(t.frame_bytes, t.raw_bytes - off);

                    size_t bound = ZSTD_compressBound(sz);
                    t.cframes[done].resize(bound);
                    size_t csize = ZSTD_compress(
                        t.cframes[done].data(), bound, t.raw + off, sz, level);
                    if (ZSTD_isError(csize)) {
                        LLAMA_LOG_ERROR("%s: ZSTD_compress failed: %s\n",
                            __func__, ZSTD_getErrorName(csize));
                        t.cframes[done].clear();
                        return;
                    }
                    t.cframes[done].resize(csize);

                    madvise(t.raw + off, sz, MADV_DONTNEED);

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

void kv_zstd_state::init(int zstd_level) {
    level     = zstd_level;
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
    // Ask the bg thread to stop at the next frame boundary.
    interrupted.store(true, std::memory_order_release);

    // Wait for it to go idle.
    {
        std::unique_lock<std::mutex> lk(mu);
        idle_cv.wait(lk, [this]{ return bg_idle; });
    }

    // Restore any frames that were compressed + MADV_DONTNEED'd.
    size_t bytes_restored = 0;
    for (auto & t : tensors) {
        int32_t done = t.n_done.load(std::memory_order_acquire);
        for (int32_t i = 0; i < done; i++) {
            size_t off = (size_t)i * t.frame_bytes;
            size_t sz  = std::min(t.frame_bytes, t.raw_bytes - off);
            assert(!t.cframes[i].empty());
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
