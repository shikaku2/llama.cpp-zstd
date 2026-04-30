#pragma once

#ifdef GGML_USE_ZSTD

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

struct kv_zstd_cell_range {
    uint32_t stream = 0;
    uint32_t begin  = 0;
    uint32_t end    = 0;
};

enum class kv_zstd_tensor_layout {
    SLOT_MAJOR,
    TRANSPOSED_V,
};

struct kv_zstd_tensor {
    uint8_t * raw         = nullptr;
    size_t    raw_bytes   = 0;
    size_t    frame_bytes = 0;
    int32_t   n_frames    = 0;
    uint32_t  n_ctx       = 0;
    uint32_t  n_stream    = 1;
    uint32_t  n_embd      = 0;
    size_t    bytes_per_slot   = 0;
    size_t    bytes_per_el     = 0;
    size_t    bytes_per_stream = 0;
    kv_zstd_tensor_layout layout = kv_zstd_tensor_layout::SLOT_MAJOR;

    std::vector<std::vector<uint8_t>> cframes;
    std::vector<int32_t> active_frames;
    size_t covered_bytes     = 0;
    size_t actual_used_bytes = 0;
    std::atomic<int32_t> n_done{0};

    kv_zstd_tensor() = default;
    kv_zstd_tensor(
            uint8_t * raw,
            size_t raw_bytes,
            size_t frame_bytes,
            uint32_t n_ctx,
            uint32_t n_stream,
            uint32_t n_embd,
            size_t bytes_per_slot,
            size_t bytes_per_el,
            kv_zstd_tensor_layout layout);

    kv_zstd_tensor(kv_zstd_tensor && o) noexcept
        : raw(o.raw), raw_bytes(o.raw_bytes), frame_bytes(o.frame_bytes), n_frames(o.n_frames),
          n_ctx(o.n_ctx), n_stream(o.n_stream), n_embd(o.n_embd), bytes_per_slot(o.bytes_per_slot),
          bytes_per_el(o.bytes_per_el), bytes_per_stream(o.bytes_per_stream), layout(o.layout),
          cframes(std::move(o.cframes)), active_frames(std::move(o.active_frames)),
          covered_bytes(o.covered_bytes), actual_used_bytes(o.actual_used_bytes),
          n_done(o.n_done.load(std::memory_order_relaxed)) {}

    kv_zstd_tensor & operator=(kv_zstd_tensor && o) noexcept {
        raw = o.raw; raw_bytes = o.raw_bytes; frame_bytes = o.frame_bytes; n_frames = o.n_frames;
        n_ctx = o.n_ctx; n_stream = o.n_stream; n_embd = o.n_embd; bytes_per_slot = o.bytes_per_slot;
        bytes_per_el = o.bytes_per_el; bytes_per_stream = o.bytes_per_stream; layout = o.layout;
        cframes = std::move(o.cframes); active_frames = std::move(o.active_frames);
        covered_bytes = o.covered_bytes; actual_used_bytes = o.actual_used_bytes;
        n_done.store(o.n_done.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }

    void select_frames(const std::vector<kv_zstd_cell_range> & ranges);
};

struct kv_zstd_state {
    std::vector<kv_zstd_tensor> tensors;
    int   level     = 1;
    float threshold = 1.00f;

    uint32_t n_used = 0;

    void init(int zstd_level, float thresh);
    void sync();
    void start(std::vector<kv_zstd_cell_range> ranges, uint32_t n_used);

    size_t compressed_bytes() const;
    size_t covered_bytes()    const;
    size_t raw_used_bytes()   const;

    ~kv_zstd_state();

private:
    std::thread             bg_thread;
    std::mutex              mu;
    std::condition_variable wake_cv;
    std::condition_variable idle_cv;
    bool                    work_ready = false;
    bool                    bg_idle    = true;
    bool                    bg_stop    = false;
    std::atomic<bool>       interrupted{false};
    std::vector<kv_zstd_cell_range> work_ranges;

    void bg_loop();
};

#endif // GGML_USE_ZSTD
