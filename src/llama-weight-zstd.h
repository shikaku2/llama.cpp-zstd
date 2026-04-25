#pragma once

#ifdef GGML_USE_ZSTD

#include "llama.h"
#include "ggml.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct llama_model;


// Per-tensor compressed blob.  original_size == 0 means the tensor compressed poorly and is
// stored raw (uncompressed) in data — the hot cache still handles it uniformly.
struct llama_tensor_compressed {
    std::vector<uint8_t> data;
    size_t               original_size; // raw byte size; 0 = stored uncompressed
    ggml_type            type;
    int64_t              ne[GGML_MAX_DIMS];
    size_t               nb[GGML_MAX_DIMS];
};

using llama_compressed_weight_map = std::unordered_map<std::string, llama_tensor_compressed>;

// Model-level state owned by llama_model.
struct llama_zstd_model_state {
    llama_compressed_weight_map weights;
    size_t bytes_before = 0;
    size_t bytes_after  = 0;
};

// Forward-declare the opaque seekable decompressor type.
struct ZSTD_seekable_s;

// Partially-decompressed tensor entry.  Stores a full-size buffer but only
// decompresses the seekable frames that are actually needed (e.g. by GET_ROWS).
// A bitset tracks which frames have been materialised so subsequent accesses
// only decompress the delta.
struct llama_zstd_partial_entry {
    std::vector<uint8_t> buf;              // full original_size allocation
    std::vector<bool>    frame_present;    // frame_present[i] == true ⇒ frame i is decompressed
    size_t               original_size;
    size_t               decompressed_bytes; // sum of decompressed frame sizes (for stats)
};

// LRU cache of decompressed tensor buffers, bounded by max_bytes.
// Sized to bytes_saved so total RAM = compressed + cache ≈ original.
struct llama_zstd_lru_cache {
    size_t max_bytes  = 0;
    size_t used_bytes = 0;

    std::list<std::string>                                             order; // front = MRU
    std::unordered_map<std::string, std::list<std::string>::iterator> iters;
    std::unordered_map<std::string, std::vector<uint8_t>>             data;

    // Returns pointer to cached data (promotes to MRU), or nullptr if not present.
    uint8_t * get(const std::string & name);

    // Evict LRU entries until there is room for `sz` bytes, returning the last
    // evicted buffer (recycled allocation) or an empty vector if none was evicted.
    // Entries whose names appear in `pinned` are skipped (never evicted); if all
    // remaining entries are pinned, the cache is allowed to exceed max_bytes.
    // Call before decompression so the evicted memory can be reused as the target buffer.
    std::vector<uint8_t> alloc(size_t sz, const std::unordered_set<std::string> * pinned = nullptr);

    // Register a filled buffer under `name`. Must be called after alloc()+decompress.
    // Returns pointer to stored data.
    uint8_t * commit(const std::string & name, std::vector<uint8_t> buf);
};

// Persistent worker pool for parallel LRU-path decompression.  Spawning std::thread
// on every forward pass costs ~50-100 µs per thread × hardware_concurrency() and
// eats the parallelism gain; reusing workers across calls avoids that.
struct llama_zstd_worker_pool {
    std::vector<std::thread>          workers;
    std::mutex                        mu;
    std::condition_variable           cv_work;   // signal workers
    std::condition_variable           cv_done;   // signal submitter
    std::function<void(size_t)>       job;       // invoked with job index
    size_t                            n_jobs     = 0;
    std::atomic<size_t>               next_job{0};
    std::atomic<size_t>               done_count{0};
    bool                              shutdown   = false;

    void start(unsigned n_threads);
    void stop();
    // Run `fn` on indices [0, n). Blocks until all jobs complete.
    // The caller thread also participates as a worker.
    void run(size_t n, std::function<void(size_t)> fn);

    ~llama_zstd_worker_pool() { stop(); }
};

// Context-level state owned by llama_context — one per inference context.
// Compressed data lives in the model; this holds the seekable decompressors
// and the bounded hot cache of recently-decompressed tensors.
struct llama_zstd_ctx_state {
    // One seekable decompressor per compressed tensor (kept for re-decompression on eviction).
    std::unordered_map<std::string, ZSTD_seekable_s *> seekables;
    // Hot LRU cache — bounded to bytes_saved so total RAM stays flat.
    llama_zstd_lru_cache cache;

    // Partial decompression entries for tensors used only in GET_ROWS.
    // These live outside the LRU cache — they are full-size allocations
    // with only the needed frames populated, so they save decompression
    // time rather than memory.  Evicted when the tensor leaves the
    // partial-eligible set (e.g. used in a matmul in a later graph).
    std::unordered_map<std::string, llama_zstd_partial_entry> partial;

    // Streaming MUL_MAT path: per-tensor traits registered as an extra_buffer_type.
    // compute_forward decompresses frame-by-frame into a thread-local L2-resident
    // buffer and computes the matmul in-place, bypassing the LRU cache entirely.
    // Owned opaque pointers (zstd_streaming_traits*) — deleted in destructor.
    std::vector<void *>              streaming_traits_ptrs;
    // Weight tensors whose ->extra was set; cleared on destruction.
    std::vector<struct ggml_tensor *> streaming_tensors;

    // Persistent pool for parallel LRU-path decompression.
    llama_zstd_worker_pool pool;

    // Temporary decompressed buffers for the current forward pass when
    // max_bytes == 0 (no-cache mode).  Cleared at the start of each
    // decompress_graph call so they stay alive exactly one pass.
    std::vector<std::vector<uint8_t>> nocache_bufs;

    ~llama_zstd_ctx_state();
};

// Compress CPU-backend tensors in model after load_tensors().
// Populates model.zstd_state.  No-op if params.cpu_weight_zstd_level == 0.
void llama_weight_zstd_compress(struct llama_model & model, const struct llama_model_params & params);

// Create context-level state (seekable decompressors + hot cache) from the model state.
// model is used to look up tensor pointers for the streaming MUL_MAT path.
// Returns nullptr if model has no compressed weights.
llama_zstd_ctx_state * llama_zstd_ctx_init(const llama_zstd_model_state & model_state, const struct llama_model & model);

// Walk all nodes in a compute graph, decompress any compressed source tensors
// not currently in the hot cache, and patch tensor->data to the cache buffer.
// Must be called single-threaded before ggml_backend_sched_graph_compute_async.
void llama_zstd_decompress_graph(
        struct ggml_cgraph          * gf,
        llama_compressed_weight_map & weights,
        llama_zstd_ctx_state        & ctx);

#endif // GGML_USE_ZSTD
