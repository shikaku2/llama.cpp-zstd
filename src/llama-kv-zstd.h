#pragma once

#ifdef GGML_USE_ZSTD

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

// Per-tensor state for async KV cache compression.
//
// The raw buffer (pointing into the ggml backend buffer) is divided into
// equal-sized frames.  The background thread compresses one frame at a time,
// storing the result in cframes[i] and issuing MADV_DONTNEED on the
// corresponding raw pages.  The atomic n_done counter is the contract
// between the bg thread (writer) and the main thread (reader):
//
//   n_done in [0,         n_frames):   first-pass frames [0, n_done) compressed
//   n_done in [n_frames,  2*n_frames): all first-pass done; second-pass frames
//                                      [0, n_done-n_frames) recompressed
//
// cframes[i] always holds the best compressed copy for frame i (or is empty if
// the frame was skipped because it didn't meet the threshold).
//
// Invariant maintained by kv_zstd_state::sync(): before each decode the
// whole raw buffer is valid (n_done == 0).
struct kv_zstd_tensor {
    uint8_t * raw        = nullptr;
    size_t    raw_bytes  = 0;
    size_t    frame_bytes = 0;
    int32_t   n_frames   = 0;
    uint32_t  n_ctx      = 0;  // total cache slots (used to compute bytes_per_token)

    std::vector<std::vector<uint8_t>> cframes;
    std::atomic<int32_t> n_done{0};

    kv_zstd_tensor() = default;
    kv_zstd_tensor(uint8_t * raw, size_t raw_bytes, size_t frame_bytes, uint32_t n_ctx);

    // Move only ever happens before the bg thread starts, so a value-copy of the atomic is safe.
    kv_zstd_tensor(kv_zstd_tensor && o) noexcept
        : raw(o.raw), raw_bytes(o.raw_bytes), frame_bytes(o.frame_bytes), n_frames(o.n_frames),
          n_ctx(o.n_ctx), cframes(std::move(o.cframes)),
          n_done(o.n_done.load(std::memory_order_relaxed)) {}

    kv_zstd_tensor & operator=(kv_zstd_tensor && o) noexcept {
        raw = o.raw; raw_bytes = o.raw_bytes; frame_bytes = o.frame_bytes; n_frames = o.n_frames;
        n_ctx = o.n_ctx; cframes = std::move(o.cframes);
        n_done.store(o.n_done.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }
};

// Owns the background compression thread and the collection of per-tensor states.
//
// Lifecycle:
//   1. Populate tensors[], then call init(level).
//   2. After each decode: call start() — bg thread begins compressing.
//   3. Before each decode: call sync() — cancels in-progress compression,
//      restores any MADV_DONTNEED'd frames, resets counters.
//   4. Destructor shuts down the bg thread cleanly.
//
// Two-pass mode (recompress_level > 0):
//   The bg thread makes two passes over each tensor.  n_done counts from 0 to
//   2*n_frames: frames [0, n_frames) are the first pass (at `level`), frames
//   [n_frames, 2*n_frames) are the second pass (at `recompress_level`).
//   sync() caps at n_frames when deciding which cframes to decompress.
struct kv_zstd_state {
    std::vector<kv_zstd_tensor> tensors;
    int   level            = 1;
    float threshold        = 1.00f; // skip frame if compressed/raw ratio exceeds this (1.0 = always compress)
    int   recompress_level = 0;     // 0 = single pass; 1-19 = second-pass target level

    uint32_t n_used = 0;  // set by start(): highest used slot index + 1

    void init(int zstd_level, float thresh, int recompress = 0);
    void sync();                   // must be called before decode: ensures raw buffers are valid
    void start(uint32_t n_used);   // must be called after decode: signals bg thread

    // Sum of compressed frame sizes across all tensors.
    // This is the real "compressed KV data size" — independent of MADV_DONTNEED
    // and OS-level zero-page accounting.
    size_t compressed_bytes() const;
    size_t raw_used_bytes()   const;  // n_used * sum(bytes_per_token)

    ~kv_zstd_state();

private:
    std::thread             bg_thread;
    std::mutex              mu;
    std::condition_variable wake_cv;  // bg waits here for work or stop
    std::condition_variable idle_cv;  // main waits here until bg is idle
    bool                    work_ready = false;
    bool                    bg_idle    = true;
    bool                    bg_stop    = false;
    std::atomic<bool>       interrupted{false};

    void bg_loop();
};

#endif // GGML_USE_ZSTD
