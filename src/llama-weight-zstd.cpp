#ifdef GGML_USE_ZSTD

#include "llama-weight-zstd.h"
#include "llama-model.h"
#include "llama-impl.h"

#include "ggml.h"
#include "../vendor/zstd-seekable/zstd_seekable.h"

// Internal ggml-cpu headers for the extra_buffer_type streaming hook
#include "traits.h"
#include "vec.h"

#include <algorithm>
#include <cstring>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>

#ifdef __linux__
#include <sys/mman.h>
#include <stdio.h>
#endif

// Read current RSS and peak RSS (high water mark) in MB.  Returns {0,0} if unavailable.
// Peak shows the largest footprint reached (e.g. during compression when both raw
// and compressed buffers are live); current shows steady-state usage.
struct rss_stats { size_t cur_mb; size_t peak_mb; };

static rss_stats read_rss() {
    rss_stats r{0, 0};
#ifdef __linux__
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) return r;
    char line[128];
    while (fgets(line, sizeof(line), f)) {
        unsigned long kb = 0;
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%lu", &kb);
            r.cur_mb = kb / 1024;
        } else if (strncmp(line, "VmHWM:", 6) == 0) {
            sscanf(line + 6, "%lu", &kb);
            r.peak_mb = kb / 1024;
        }
    }
    fclose(f);
#endif
    return r;
}

// ---------------------------------------------------------------------------
// Streaming MUL_MAT path (cache-resident decompression)
// ---------------------------------------------------------------------------
//
// For F16 weight tensors, instead of decompressing the whole tensor to a RAM
// buffer before GGML's kernel runs, we decompress one seekable frame (~256 KB)
// at a time into a thread-local buffer that fits in L2 cache, compute the dot
// products for that frame's rows immediately, then reuse the buffer for the
// next frame.  The decompressed data never needs to be written back to RAM.
//
// Total RAM traffic per tensor = compressed size (reads only), vs the old
// approach of compressed reads + decompressed writes + decompressed reads.
// This is the "cache-resident decompression pipeline" pattern.
//
// Integration: we register a ggml::cpu::extra_buffer_type so that
// ggml_cpu_extra_compute_forward() intercepts each MUL_MAT node whose src[0]
// weight tensor has our traits pointer in ->extra.  At that point src1
// (activations) are already computed by earlier graph nodes.

namespace {

// Per-thread cache of ZSTD seekable handles, keyed by the compressed buffer pointer
// (one buffer per tensor).  Init is cheap (parses seek table only) but doing it
// every compute_forward call adds up; cache instead.  Destructor frees handles
// when the thread exits (e.g. when the threadpool is destroyed).
struct per_thread_seekable_cache {
    std::unordered_map<const uint8_t *, ZSTD_seekable *> map;
    ~per_thread_seekable_cache() {
        for (auto & kv : map) {
            if (kv.second) ZSTD_seekable_free(kv.second);
        }
    }
};

static ZSTD_seekable * get_thread_seekable(const llama_tensor_compressed * tc) {
    thread_local per_thread_seekable_cache cache;
    const uint8_t * key = tc->data.data();
    auto it = cache.map.find(key);
    if (it != cache.map.end()) return it->second;

    ZSTD_seekable * zs = ZSTD_seekable_create();
    if (!zs) return nullptr;
    size_t r = ZSTD_seekable_initBuff(zs, tc->data.data(), tc->data.size());
    if (ZSTD_isError(r)) {
        ZSTD_seekable_free(zs);
        return nullptr;
    }
    cache.map[key] = zs;
    return zs;
}

struct zstd_streaming_traits : public ggml::cpu::tensor_traits {
    // Init-time fields (all set by llama_zstd_ctx_init):
    const llama_tensor_compressed * tc        = nullptr;
    size_t                          row_bytes = 0;   // ne00 * sizeof(ggml_fp16_t)
    int64_t                         ne00      = 0;
    int64_t                         ne01      = 0;
    int64_t                         ne02      = 1;   // n_experts (1 for plain MUL_MAT)
    size_t                          nb02      = 0;   // bytes per expert (= ne01 * row_bytes)
    bool                            aligned   = false; // frames divide cleanly into rows
    bool                            is_3d     = false; // true → MUL_MAT_ID-eligible

    // Set by decompress_graph when this tensor is hot in the LRU cache and
    // src0->data has been patched to the decompressed buffer.  Cleared at the
    // start of each decompress_graph call so an eviction is detected.
    // Non-null → compute_forward returns false immediately; GGML's AVX2/FMA
    // matmul kernel runs on the pre-decompressed data.
    const uint8_t * preloaded = nullptr;

    bool work_size(int, const struct ggml_tensor *, size_t & sz) override {
        sz = 0;
        return false;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override;

    // Streaming kernel for MUL_MAT_ID (MoE expert weights).  Decompresses only
    // the frames covering active experts; bypasses the full ~10 GB allocation.
    bool compute_forward_id(struct ggml_compute_params * params, struct ggml_tensor * op);
};

struct zstd_extra_buf_type : public ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        if (!op->src[0] || op->src[0]->extra == nullptr) return false;
        if (op->src[0]->type != GGML_TYPE_F16)           return false;
        return op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (!op->src[0] || op->src[0]->extra == nullptr) return nullptr;
        if (op->src[0]->type != GGML_TYPE_F16)           return nullptr;
        if (op->op != GGML_OP_MUL_MAT && op->op != GGML_OP_MUL_MAT_ID) return nullptr;
        return static_cast<ggml::cpu::tensor_traits *>(op->src[0]->extra);
    }
};

bool zstd_streaming_traits::compute_forward(
        struct ggml_compute_params * params, struct ggml_tensor * op) {

    if (op->op == GGML_OP_MUL_MAT_ID) {
        return compute_forward_id(params, op);
    }

    struct ggml_tensor * src0 = op->src[0];  // F16 weights
    struct ggml_tensor * src1 = op->src[1];  // F32 activations

    const int ith = params->ith;
    const int nth = params->nth;

    // Fast path: decompress_graph already decompressed this tensor into the LRU
    // cache and patched src0->data.  Let GGML's optimised AVX2/FMA matmul run.
    if (preloaded != nullptr) {
        return false;
    }

    // Timing: thread 0 measures decompression vs compute time.
    // Other threads run in parallel — thread 0 is representative of the split.
    int64_t t0_cf = 0;
    if (ith == 0) t0_cf = ggml_time_us();

    // Runtime validation — fall back if conditions don't fit the streaming path.
    // Init-time validation already confirmed src0 is F16, contiguous, 2D, aligned.
    const bool ok = aligned
                 && src1->type == GGML_TYPE_F32
                 && ggml_is_contiguous(src1)
                 && src1->ne[2] == 1 && src1->ne[3] == 1;

    if (!ok) {
        // Thread 0 decompresses the whole tensor into src0->data, then we
        // barrier so all threads see the result, then return false to let
        // GGML run its standard kernel.
        if (ith == 0) {
            ZSTD_seekable * zs = get_thread_seekable(tc);
            if (zs) {
                size_t r = ZSTD_seekable_decompress(zs, src0->data, tc->original_size, 0);
                if (ZSTD_isError(r)) {
                    LLAMA_LOG_ERROR("zstd streaming fallback: %s\n", ZSTD_getErrorName(r));
                }
            }
        }
        ggml_barrier(params->threadpool);
        return false;
    }

    const int64_t ne11 = src1->ne[1];   // activation vectors (N, batch)

    ZSTD_seekable * zs = get_thread_seekable(tc);
    if (!zs) {
        // Should be very rare; fall back via barrier+false.
        if (ith == 0) {
            LLAMA_LOG_ERROR("zstd streaming: get_thread_seekable failed for %s\n", src0->name);
        }
        ggml_barrier(params->threadpool);
        return false;
    }
    const unsigned n_frames = ZSTD_seekable_getNumFrames(zs);

    // Per-thread F16 activation buffer.  Each thread does the conversion
    // independently — for typical batches this is microseconds, less than
    // the cost of a barrier.  The buffer stays in L1/L2 for the frame loop.
    thread_local std::vector<ggml_fp16_t> act_f16;
    act_f16.resize((size_t)ne00 * (size_t)ne11);
    for (int64_t i1 = 0; i1 < ne11; i1++) {
        const float * src = (const float *)((const char *)src1->data + i1 * src1->nb[1]);
        ggml_fp32_to_fp16_row(src, act_f16.data() + i1 * ne00, ne00);
    }

    // Per-thread frame buffer (~256 KB).  Reused across frames so it
    // stays in L2 (Zen 3 has 512 KB L2 per core).  Decompressed bytes are
    // consumed by the dot products before the next frame overwrites them,
    // so they never need to be written back to RAM.
    thread_local std::vector<uint8_t> frame_buf;

    // Each thread takes a stride of frames: thread 0 -> frames 0, nth, 2*nth, ...
    // Different frames cover different output rows, so dst writes don't collide.
    constexpr int K = GGML_VEC_DOT_UNROLL;  // 2: dots K vectors against 1 in parallel
    const int act_stride = (int)(ne00 * sizeof(ggml_fp16_t));

    int64_t t_zstd_us    = 0;  // thread-0 accumulator for ZSTD decompression
    int64_t t_compute_us = 0;  // thread-0 accumulator for dot products

    for (unsigned fi = (unsigned)ith; fi < n_frames; fi += (unsigned)nth) {
        unsigned long long f_off = ZSTD_seekable_getFrameDecompressedOffset(zs, fi);
        size_t             f_sz  = ZSTD_seekable_getFrameDecompressedSize(zs, fi);

        frame_buf.resize(f_sz);
        int64_t t_zstd0 = (ith == 0) ? ggml_time_us() : 0;
        size_t r = ZSTD_seekable_decompress(zs, frame_buf.data(), f_sz, f_off);
        int64_t t_zstd1 = (ith == 0) ? ggml_time_us() : 0;
        if (ith == 0) t_zstd_us += t_zstd1 - t_zstd0;

        if (ZSTD_isError(r)) {
            LLAMA_LOG_ERROR("zstd streaming frame %u: %s\n", fi, ZSTD_getErrorName(r));
            continue;
        }

        const int64_t row_lo = (int64_t)(f_off / row_bytes);
        const int64_t row_hi = std::min((int64_t)ne01, row_lo + (int64_t)(f_sz / row_bytes));

        int64_t t_dot0 = (ith == 0) ? ggml_time_us() : 0;
        if (ne11 == 1) {
            // Single-token (gen) path: K weight rows × 1 activation per call.
            // Activation stays in registers; pairs of weight rows feed both
            // accumulators per inner SIMD iter.  Output writes are contiguous.
            const ggml_fp16_t * act  = act_f16.data();
            float             * dcol = (float *)op->data;
            int64_t r0 = row_lo;
            for (; r0 + K <= row_hi; r0 += K) {
                const ggml_fp16_t * w_base = (const ggml_fp16_t *)
                    (frame_buf.data() + (r0 - row_lo) * row_bytes);
                float results[K];
                ggml_vec_dot_f16_unroll(
                    (int)ne00, (int)row_bytes, results,
                    const_cast<ggml_fp16_t *>(w_base),
                    const_cast<ggml_fp16_t *>(act));
                for (int k = 0; k < K; k++) {
                    dcol[r0 + k] = results[k];
                }
            }
            for (; r0 < row_hi; r0++) {
                const ggml_fp16_t * w = (const ggml_fp16_t *)
                    (frame_buf.data() + (r0 - row_lo) * row_bytes);
                float result = 0.0f;
                ggml_vec_dot_f16((int)ne00, &result, 0,
                    const_cast<ggml_fp16_t *>(w), 0,
                    const_cast<ggml_fp16_t *>(act), 0, 1);
                dcol[r0] = result;
            }
        } else {
            // Batched (prompt) path: weight row outer, K activations inner.
            // Each weight row is loaded once and reused for K dot products
            // with K accumulators in registers.  Activations stay in L1.
            for (int64_t r0 = row_lo; r0 < row_hi; r0++) {
                const ggml_fp16_t * w = (const ggml_fp16_t *)
                    (frame_buf.data() + (r0 - row_lo) * row_bytes);
                int64_t i1 = 0;
                for (; i1 + K <= ne11; i1 += K) {
                    float results[K];
                    ggml_vec_dot_f16_unroll(
                        (int)ne00, act_stride, results,
                        act_f16.data() + i1 * ne00,
                        const_cast<ggml_fp16_t *>(w));
                    for (int k = 0; k < K; k++) {
                        float * dcol = (float *)((char *)op->data + (i1 + k) * op->nb[1]);
                        dcol[r0] = results[k];
                    }
                }
                for (; i1 < ne11; i1++) {
                    const ggml_fp16_t * act = act_f16.data() + i1 * ne00;
                    float * dcol = (float *)((char *)op->data + i1 * op->nb[1]);
                    float result = 0.0f;
                    ggml_vec_dot_f16((int)ne00, &result, 0,
                        const_cast<ggml_fp16_t *>(w), 0,
                        const_cast<ggml_fp16_t *>(act), 0, 1);
                    dcol[r0] = result;
                }
            }
        }
        if (ith == 0) t_compute_us += ggml_time_us() - t_dot0;
    }

    if (ith == 0) {
        int64_t t1_cf = ggml_time_us();
        static std::atomic<uint64_t> s_cf_call{0};
        uint64_t cfcall = s_cf_call.fetch_add(1, std::memory_order_relaxed);
        // Log first 30 calls (covers ~1 token × 21 streaming tensors),
        // then every 200 to capture steady state.
        if (cfcall < 30 || cfcall % 200 == 0) {
            LLAMA_LOG_INFO("zstd stream #%zu %s (ne01=%lld ne11=%lld): "
                           "zstd=%.2fms dot=%.2fms wall=%.2fms (th0/%d, %u frames)\n",
                           (size_t)cfcall + 1, src0->name,
                           (long long)ne01, (long long)ne11,
                           t_zstd_us   / 1e3f,
                           t_compute_us / 1e3f,
                           (t1_cf - t0_cf) / 1e3f,
                           nth, n_frames);
        }
    }

    return true;
}

// Routing entry: identifies one (token-slot, token-row) pair routed to an expert.
// Layout matches ggml's mmid_row_mapping for clarity.
struct zstd_mmid_entry { int32_t i1; int32_t i2; };

// One unit of streaming work: decompress frame `fi` of the source tensor and
// run vec_dots for whichever rows fall within expert `cur_a`.  The job pool is
// flat across all (active expert, frame) pairs so threads stay busy without
// per-expert barriers.
struct zstd_mmid_job { int32_t cur_a; uint32_t fi; };

// Streaming MUL_MAT_ID kernel.  Same idea as the 2D MUL_MAT streaming above
// but applied per-active-expert: for each cur_a with cne1 > 0, the byte range
// [cur_a*nb02, (cur_a+1)*nb02) is decomposed into seekable frames distributed
// across threads.  Inactive experts are not decompressed at all — for OLMoE
// with top-8 of 64 experts, this is ~12% of the tensor instead of 100%.
//
// Activations are converted to F16 once into params->wdata (shared across
// threads, partitioned by ne11).  The routing table is built into wdata too.
bool zstd_streaming_traits::compute_forward_id(
        struct ggml_compute_params * params, struct ggml_tensor * op) {

    struct ggml_tensor * src0 = op->src[0];  // 3D F16 expert weights
    struct ggml_tensor * src1 = op->src[1];  // F32 activations
    struct ggml_tensor * ids  = op->src[2];  // I32 routing table

    const int ith = params->ith;
    const int nth = params->nth;

    // Cached path: decompress_graph already materialised src0->data; let
    // ggml's standard MUL_MAT_ID kernel run.
    if (preloaded != nullptr) {
        return false;
    }

    const int64_t ne00_l = src0->ne[0];
    const int64_t ne01_l = src0->ne[1];
    const int64_t n_as   = src0->ne[2];
    const int64_t ne10   = src1->ne[0];
    const int64_t ne11   = src1->ne[1];
    const int64_t ne12   = src1->ne[2];
    const int64_t ne13   = src1->ne[3];
    const int64_t n_ids  = ids->ne[0];
    const int64_t n_iid1 = ids->ne[1];

    // Conditions outside our streaming kernel's coverage — fall back via the
    // partial-decompression escape hatch below.  src1 must be F32 contiguous;
    // ne13==1 covers all real LLM cases (no batch-of-batches).
    const bool ok = aligned
                 && is_3d
                 && src1->type == GGML_TYPE_F32
                 && ggml_is_contiguous(src1)
                 && ne13 == 1
                 && ne00_l == ne10;

    if (!ok) {
        // Conservative fallback: thread 0 decompresses the whole tensor into
        // src0->data, barrier, and let ggml's kernel handle it.  src0->data
        // points at the original (MADV_DONTNEED'd) address range — the kernel
        // re-faults the pages on write.
        if (ith == 0) {
            ZSTD_seekable * zs = get_thread_seekable(tc);
            if (zs) {
                size_t r = ZSTD_seekable_decompress(zs, src0->data, tc->original_size, 0);
                if (ZSTD_isError(r)) {
                    LLAMA_LOG_ERROR("zstd streaming MMID fallback: %s\n", ZSTD_getErrorName(r));
                }
            }
        }
        ggml_barrier(params->threadpool);
        return false;
    }

    // ---- Activation F16 conversion ----
    // Each thread independently converts ALL rows so that the worker loop
    // below can read any row regardless of which thread processes a given job.
    // Using thread_local avoids the allocation on every call while keeping the
    // data private (no sharing needed — every thread writes all rows itself).
    thread_local std::vector<ggml_fp16_t> act_f16;
    const size_t total_act = (size_t)ne10 * (size_t)ne11 * (size_t)ne12;
    act_f16.resize(total_act);

    for (int64_t i12 = 0; i12 < ne12; ++i12) {
        for (int64_t i11 = 0; i11 < ne11; ++i11) {
            const float * s = (const float *)((const char *)src1->data
                                              + i12 * src1->nb[2]
                                              + i11 * src1->nb[1]);
            ggml_fp32_to_fp16_row(s, act_f16.data() + i12 * ne11 * ne10 + i11 * ne10, (int)ne10);
        }
    }

    // ---- Build routing table + flat jobs list (single-threaded, on ith==0) ----
    //
    // Both structures live in thread-shared statics, mutated only by ith==0
    // between barriers.  We also need a per-call seekable handle on thread 0
    // to compute frame ranges; threads use their own thread_local handles for
    // the actual decode.
    static std::mutex                       s_rt_mu;
    static std::vector<int64_t>             s_row_counts;
    static std::vector<zstd_mmid_entry>     s_row_table;
    static std::vector<zstd_mmid_job>       s_jobs;
    static std::atomic<size_t>              s_next_job{0};
    static int                              s_n_active = 0;

    int64_t t0_cf = (ith == 0) ? ggml_time_us() : 0;

    if (ith == 0) {
        std::lock_guard<std::mutex> lk(s_rt_mu);
        s_row_counts.assign((size_t)n_as, 0);
        s_row_table.assign((size_t)n_as * n_ids * n_iid1, {0, 0});
        for (int64_t iid1 = 0; iid1 < n_iid1; ++iid1) {
            for (int64_t id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *)((const char *)ids->data
                                                       + iid1 * ids->nb[1]
                                                       + id   * ids->nb[0]);
                int64_t k = s_row_counts[i02]++;
                s_row_table[(size_t)i02 * n_ids * n_iid1 + (size_t)k] =
                    { (int32_t)id, (int32_t)iid1 };
            }
        }

        // Build flat jobs list across all (active expert, frame) pairs.
        ZSTD_seekable * zs0 = get_thread_seekable(tc);
        s_jobs.clear();
        s_n_active = 0;
        if (zs0) {
            for (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
                if (s_row_counts[cur_a] == 0) continue;
                ++s_n_active;
                const size_t expert_off = (size_t)cur_a * (size_t)src0->nb[2];
                const size_t expert_end = expert_off + (size_t)src0->nb[2];
                const unsigned frame_lo = ZSTD_seekable_offsetToFrameIndex(zs0, (unsigned long long)expert_off);
                const unsigned frame_hi = ZSTD_seekable_offsetToFrameIndex(zs0, (unsigned long long)(expert_end - 1));
                for (unsigned fi = frame_lo; fi <= frame_hi; ++fi) {
                    s_jobs.push_back({ (int32_t)cur_a, fi });
                }
            }
        }
        s_next_job.store(0, std::memory_order_relaxed);
    }

    // Activation conversion + routing table + jobs list must all be visible
    // before the worker loop starts.
    ggml_barrier(params->threadpool);

    ZSTD_seekable * zs = get_thread_seekable(tc);
    if (!zs) {
        if (ith == 0) {
            LLAMA_LOG_ERROR("zstd streaming MMID: get_thread_seekable failed for %s\n", src0->name);
        }
        ggml_barrier(params->threadpool);
        return false;
    }

    int64_t t_zstd_us       = 0;
    int64_t t_compute_us    = 0;
    unsigned n_decompressed = 0;

    thread_local std::vector<uint8_t> frame_buf;

    // ---- Flat parallel job loop (no per-expert barriers) ----
    while (true) {
        size_t ji = s_next_job.fetch_add(1, std::memory_order_relaxed);
        if (ji >= s_jobs.size()) break;

        const zstd_mmid_job job = s_jobs[ji];
        const int64_t cur_a = job.cur_a;
        const unsigned fi   = job.fi;
        const int64_t cne1  = s_row_counts[cur_a];

        const size_t expert_off = (size_t)cur_a * (size_t)src0->nb[2];
        const size_t expert_end = expert_off + (size_t)src0->nb[2];

        unsigned long long f_off = ZSTD_seekable_getFrameDecompressedOffset(zs, fi);
        size_t             f_sz  = ZSTD_seekable_getFrameDecompressedSize(zs, fi);

        frame_buf.resize(f_sz);
        int64_t t_zstd0 = (ith == 0) ? ggml_time_us() : 0;
        size_t r = ZSTD_seekable_decompress(zs, frame_buf.data(), f_sz, f_off);
        if (ith == 0) {
            t_zstd_us += ggml_time_us() - t_zstd0;
            ++n_decompressed;
        }

        if (ZSTD_isError(r)) {
            LLAMA_LOG_ERROR("zstd MMID frame %u: %s\n", fi, ZSTD_getErrorName(r));
            continue;
        }

        // Overlap of frame with this expert (frame may straddle expert
        // boundary; clip to the expert's byte range).
        unsigned long long b_lo = std::max(f_off, (unsigned long long)expert_off);
        unsigned long long b_hi = std::min(f_off + f_sz, (unsigned long long)expert_end);
        if (b_lo >= b_hi) continue;

        const int64_t row_lo = (int64_t)((b_lo - expert_off) / row_bytes);
        int64_t       row_hi = (int64_t)((b_hi - expert_off) / row_bytes);
        if (row_hi > ne01_l) row_hi = ne01_l;

        const zstd_mmid_entry * exp_rows = &s_row_table[(size_t)cur_a * n_ids * n_iid1];

        int64_t t_dot0 = (ith == 0) ? ggml_time_us() : 0;
        for (int64_t row = row_lo; row < row_hi; ++row) {
            const size_t buf_offset =
                (size_t)((unsigned long long)expert_off + (unsigned long long)row * row_bytes - f_off);
            const ggml_fp16_t * w = (const ggml_fp16_t *)(frame_buf.data() + buf_offset);

            for (int64_t k = 0; k < cne1; ++k) {
                const zstd_mmid_entry m = exp_rows[k];
                const int64_t i11 = m.i1 % ne11;
                const int64_t i12 = m.i2;

                const ggml_fp16_t * act = act_f16.data() + i12 * ne11 * ne10 + i11 * ne10;

                float result = 0.0f;
                ggml_vec_dot_f16((int)ne00_l, &result, 0,
                    const_cast<ggml_fp16_t *>(w), 0,
                    const_cast<ggml_fp16_t *>(act), 0, 1);

                float * dst_col = (float *)((char *)op->data
                                            + (int64_t)m.i1 * op->nb[1]
                                            + (int64_t)m.i2 * op->nb[2]);
                dst_col[row] = result;
            }
        }
        if (ith == 0) t_compute_us += ggml_time_us() - t_dot0;
    }

    // Single barrier: ensure every thread's dst writes are visible to the
    // next op in the graph.  No per-expert barriers anymore — dst writes for
    // different (cur_a, k) pairs land in disjoint (i1, i2) cells (each
    // routing entry is unique), so no inter-thread write conflicts.
    ggml_barrier(params->threadpool);

    if (ith == 0) {
        int64_t t1_cf = ggml_time_us();
        static std::atomic<uint64_t> s_cf_call{0};
        uint64_t cfcall = s_cf_call.fetch_add(1, std::memory_order_relaxed);
        if (cfcall < 30 || cfcall % 200 == 0) {
            LLAMA_LOG_INFO("zstd MMID #%zu %s (n_as=%lld active=%d jobs=%zu th0_frames=%u): "
                           "zstd=%.2fms dot=%.2fms wall=%.2fms (th0/%d)\n",
                           (size_t)cfcall + 1, src0->name,
                           (long long)n_as, s_n_active, s_jobs.size(), n_decompressed,
                           t_zstd_us    / 1e3f,
                           t_compute_us / 1e3f,
                           (t1_cf - t0_cf) / 1e3f,
                           nth);
        }
    }

    return true;
}

// Global singleton extra-buffer-type.  Its context pointer points to the
// extra_buffer_type instance; that's all ggml_cpu_extra_compute_forward needs.
static zstd_extra_buf_type g_zstd_ebt;
static const char * zstd_buft_name(ggml_backend_buffer_type_t) { return "ZSTD-STREAMING"; }
static ggml_backend_buffer_type_i g_zstd_buft_iface = {
    /* .get_name       = */ zstd_buft_name,
    /* .alloc_buffer   = */ nullptr,
    /* .get_alignment  = */ nullptr,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ nullptr,
};
static ggml_backend_buffer_type g_zstd_buft = {
    g_zstd_buft_iface,
    /* device  = */ nullptr,
    /* context = */ &g_zstd_ebt,
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Compression
// ---------------------------------------------------------------------------

static std::vector<uint8_t> compress_seekable(
        const void * src, size_t src_size,
        int level, unsigned frame_kb) {

    ZSTD_seekable_CStream * zcs = ZSTD_seekable_createCStream();
    if (!zcs) {
        throw std::runtime_error("ZSTD_seekable_createCStream failed");
    }

    unsigned frame_bytes = (unsigned)(frame_kb * 1024u);
    size_t ret = ZSTD_seekable_initCStream(zcs, level, /*checksumFlag=*/0, frame_bytes);
    if (ZSTD_isError(ret)) {
        ZSTD_seekable_freeCStream(zcs);
        throw std::runtime_error(std::string("ZSTD_seekable_initCStream: ") + ZSTD_getErrorName(ret));
    }

    // Worst-case bound: slightly larger than input
    size_t bound = ZSTD_compressBound(src_size) + 1024 * 1024; // extra for seek table
    std::vector<uint8_t> dst(bound);

    ZSTD_inBuffer  in  = { src, src_size, 0 };
    ZSTD_outBuffer out = { dst.data(), dst.size(), 0 };

    while (in.pos < in.size) {
        size_t r = ZSTD_seekable_compressStream(zcs, &out, &in);
        if (ZSTD_isError(r)) {
            ZSTD_seekable_freeCStream(zcs);
            throw std::runtime_error(std::string("ZSTD_seekable_compressStream: ") + ZSTD_getErrorName(r));
        }
    }

    // Flush and write seek table
    size_t r;
    do {
        r = ZSTD_seekable_endStream(zcs, &out);
        if (ZSTD_isError(r)) {
            ZSTD_seekable_freeCStream(zcs);
            throw std::runtime_error(std::string("ZSTD_seekable_endStream: ") + ZSTD_getErrorName(r));
        }
        if (r > 0 && out.pos == out.size) {
            // Need more space (shouldn't happen with our bound, but be safe)
            dst.resize(dst.size() * 2);
            out.dst  = dst.data();
            out.size = dst.size();
        }
    } while (r != 0);

    ZSTD_seekable_freeCStream(zcs);

    dst.resize(out.pos);
    dst.shrink_to_fit();
    return dst;
}

void llama_weight_zstd_compress(struct llama_model & model, const struct llama_model_params & params) {
    if (params.cpu_weight_zstd_level <= 0) {
        return;
    }

    if (!model.zstd_state) {
        model.zstd_state = std::make_unique<llama_zstd_model_state>();
    }
    auto & state = *model.zstd_state;

    const int   level      = params.cpu_weight_zstd_level;
    const float threshold  = params.cpu_weight_zstd_threshold;
    const int   frame_kb   = params.cpu_weight_zstd_frame_kb > 0 ? params.cpu_weight_zstd_frame_kb : 32;
    const bool  validate   = params.cpu_weight_zstd_validate;

    LLAMA_LOG_INFO("%s: compressing CPU weights with zstd level=%d threshold=%.2f frame_kb=%d\n",
                   __func__, level, threshold, frame_kb);

    // Collect eligible tensors (CPU host buffers, has data, has a name)
    std::vector<struct ggml_tensor *> candidates;
    int n_gpu = 0, n_no_data = 0;
    size_t sz_cpu = 0, sz_gpu = 0;
    for (auto & [name, tensor] : model.tensors_by_name) {
        if (!tensor || !tensor->data || !tensor->buffer) {
            ++n_no_data;
            continue;
        }
        if (!ggml_backend_buffer_is_host(tensor->buffer)) {
            ++n_gpu;
            sz_gpu += ggml_nbytes(tensor);
            continue;
        }
        size_t sz = ggml_nbytes(tensor);
        if (sz == 0) {
            continue;
        }
        sz_cpu += sz;
        candidates.push_back(tensor);
    }

    LLAMA_LOG_INFO("%s: tensor placement: %d on CPU (%.1f MB), %d on GPU/other (%.1f MB), %d no-data\n",
                   __func__,
                   (int)candidates.size(), sz_cpu / 1e6,
                   n_gpu, sz_gpu / 1e6,
                   n_no_data);

    if (candidates.empty()) {
        LLAMA_LOG_INFO("%s: no eligible CPU tensors found\n", __func__);
        return;
    }

    const int n_threads = params.cpu_weight_zstd_threads > 0
            ? params.cpu_weight_zstd_threads
            : std::max(1, (int)std::thread::hardware_concurrency());

    size_t bytes_before = 0;
    size_t bytes_after  = 0;
    int    n_compressed = 0;
    int    n_skipped    = 0;

    // Carry tensor pointer alongside future so we can MADV_DONTNEED each tensor's
    // original pages as soon as its compressed copy is stored — before the next
    // batch starts.  This keeps peak RSS near the uncompressed baseline (~13 GB)
    // instead of spiking to uncompressed + all-compressed (~25 GB).
    struct compress_job {
        struct ggml_tensor *                    tensor;
        std::future<llama_tensor_compressed>    future;
    };
    std::vector<compress_job> jobs;
    jobs.reserve((size_t)n_threads);

    // Process in batches of n_threads; MADV_DONTNEED each tensor immediately
    // after its result is collected so its physical pages are reclaimed before
    // the next batch allocates compressed output.
    size_t batch_start = 0;
    while (batch_start < candidates.size()) {
        size_t batch_end = std::min(batch_start + (size_t)n_threads, candidates.size());
        for (size_t i = batch_start; i < batch_end; ++i) {
            struct ggml_tensor * t = candidates[i];
            const void * data      = t->data;
            size_t       sz        = ggml_nbytes(t);
            ggml_type    type      = t->type;
            int64_t      ne[GGML_MAX_DIMS];
            size_t       nb[GGML_MAX_DIMS];
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                ne[d] = t->ne[d];
                nb[d] = t->nb[d];
            }

            jobs.push_back({t, std::async(std::launch::async,
                [data, sz, type, ne_copy = std::vector<int64_t>(ne, ne+GGML_MAX_DIMS),
                 nb_copy = std::vector<size_t>(nb, nb+GGML_MAX_DIMS),
                 level, frame_kb, threshold, validate]() mutable -> llama_tensor_compressed {
                    llama_tensor_compressed tc;
                    tc.type          = type;
                    tc.original_size = sz;
                    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                        tc.ne[d] = ne_copy[d];
                        tc.nb[d] = nb_copy[d];
                    }

                    auto compressed = compress_seekable(data, sz, level, (unsigned)frame_kb);

                    float ratio = (float)compressed.size() / (float)sz;
                    if (ratio > threshold) {
                        // Poor compression — store raw
                        tc.data          = std::vector<uint8_t>((const uint8_t *)data, (const uint8_t *)data + sz);
                        tc.original_size = 0; // sentinel: not compressed
                    } else {
                        tc.data = std::move(compressed);

                        if (validate) {
                            // Round-trip check
                            ZSTD_seekable * zs = ZSTD_seekable_create();
                            if (!zs) throw std::runtime_error("validation: ZSTD_seekable_create failed");
                            size_t r = ZSTD_seekable_initBuff(zs, tc.data.data(), tc.data.size());
                            if (ZSTD_isError(r)) {
                                ZSTD_seekable_free(zs);
                                throw std::runtime_error(std::string("validation: initBuff: ") + ZSTD_getErrorName(r));
                            }
                            std::vector<uint8_t> check(sz);
                            r = ZSTD_seekable_decompress(zs, check.data(), sz, 0);
                            ZSTD_seekable_free(zs);
                            if (ZSTD_isError(r)) {
                                throw std::runtime_error(std::string("validation: decompress: ") + ZSTD_getErrorName(r));
                            }
                            if (memcmp(check.data(), data, sz) != 0) {
                                throw std::runtime_error("validation: round-trip mismatch");
                            }
                        }
                    }
                    return tc;
                })});
        }

        // Collect this batch; MADV_DONTNEED each tensor immediately after storing
        // its compressed copy so physical pages are reclaimed before the next batch.
        for (auto & job : jobs) {
            const std::string name = job.tensor->name;
            try {
                auto tc = job.future.get();
                bytes_before += tc.original_size > 0 ? tc.original_size : tc.data.size();
                bytes_after  += tc.data.size();
                if (tc.original_size > 0) {
                    ++n_compressed;
                } else {
                    ++n_skipped;
                }
                state.weights[name] = std::move(tc);
            } catch (const std::exception & e) {
                LLAMA_LOG_WARN("%s: compression failed for tensor %s: %s\n", __func__, name.c_str(), e.what());
            }

#ifdef __linux__
            // Release original pages immediately — compressed copy is now stored.
            if (state.weights.count(name)) {
                static const size_t PAGE = 4096;
                uintptr_t addr    = (uintptr_t)job.tensor->data;
                uintptr_t aligned = addr & ~(PAGE - 1);
                size_t    sz      = ggml_nbytes(job.tensor) + (addr - aligned);
                madvise((void *)aligned, sz, MADV_DONTNEED);
            }
#endif
        }
        jobs.clear();
        batch_start = batch_end;
    }

    state.bytes_before = bytes_before;
    state.bytes_after  = bytes_after;

    {
        rss_stats rss = read_rss();
        LLAMA_LOG_INFO("%s: compressed %d tensors (%.1f MB → %.1f MB, ratio %.2f), skipped %d | rss=%zu MB peak=%zu MB\n",
                       __func__,
                       n_compressed,
                       bytes_before / 1e6,
                       bytes_after  / 1e6,
                       bytes_before > 0 ? (float)bytes_after / (float)bytes_before : 1.0f,
                       n_skipped,
                       rss.cur_mb, rss.peak_mb);
    }
}

// ---------------------------------------------------------------------------
// Context-level state
// ---------------------------------------------------------------------------

uint8_t * llama_zstd_lru_cache::get(const std::string & name) {
    auto it = data.find(name);
    if (it == data.end()) return nullptr;
    order.erase(iters.at(name));
    order.push_front(name);
    iters[name] = order.begin();
    return it->second.data();
}

std::vector<uint8_t> llama_zstd_lru_cache::alloc(size_t sz, const std::unordered_set<std::string> * pinned) {
    std::vector<uint8_t> recycled;
    while (used_bytes + sz > max_bytes && !order.empty()) {
        // Walk LRU tail toward head looking for an unpinned victim.
        auto victim = order.end();
        for (auto it = std::prev(order.end()); ; --it) {
            if (!pinned || !pinned->count(*it)) {
                victim = it;
                break;
            }
            if (it == order.begin()) break;
        }
        if (victim == order.end()) break; // all entries pinned — let cache grow

        const std::string name = *victim; // copy: erasing invalidates the reference
        used_bytes -= data.at(name).size();
        if (recycled.empty()) {
            recycled = std::move(data.at(name)); // take memory, avoid free
        }
        data.erase(name);
        iters.erase(name);
        order.erase(victim);
    }
    recycled.resize(sz); // reuse allocation if big enough, else realloc
    return recycled;
}

uint8_t * llama_zstd_lru_cache::commit(const std::string & name, std::vector<uint8_t> buf) {
    size_t sz = buf.size();
    order.push_front(name);
    auto [it, ok] = data.emplace(name, std::move(buf));
    (void)ok;
    iters[name] = order.begin();
    used_bytes += sz;
    return it->second.data();
}

void llama_zstd_worker_pool::start(unsigned n_threads) {
    if (!workers.empty()) return;
    shutdown = false;
    for (unsigned t = 0; t < n_threads; ++t) {
        workers.emplace_back([this]() {
            while (true) {
                std::unique_lock<std::mutex> lk(mu);
                cv_work.wait(lk, [this]() { return shutdown || n_jobs > 0; });
                if (shutdown && n_jobs == 0) return;
                lk.unlock();
                // Pull jobs from the atomic queue.
                while (true) {
                    size_t i = next_job.fetch_add(1, std::memory_order_relaxed);
                    if (i >= n_jobs) break;
                    job(i);
                    if (done_count.fetch_add(1, std::memory_order_acq_rel) + 1 == n_jobs) {
                        std::lock_guard<std::mutex> dlk(mu);
                        cv_done.notify_one();
                    }
                }
            }
        });
    }
}

void llama_zstd_worker_pool::stop() {
    if (workers.empty()) return;
    {
        std::lock_guard<std::mutex> lk(mu);
        shutdown = true;
    }
    cv_work.notify_all();
    for (auto & t : workers) t.join();
    workers.clear();
}

void llama_zstd_worker_pool::run(size_t n, std::function<void(size_t)> fn) {
    if (n == 0) return;
    if (workers.empty()) {
        // No pool — run serially on caller.
        for (size_t i = 0; i < n; ++i) fn(i);
        return;
    }

    {
        std::lock_guard<std::mutex> lk(mu);
        job        = std::move(fn);
        n_jobs     = n;
        next_job.store(0, std::memory_order_relaxed);
        done_count.store(0, std::memory_order_relaxed);
    }
    cv_work.notify_all();

    // Caller thread also participates.
    while (true) {
        size_t i = next_job.fetch_add(1, std::memory_order_relaxed);
        if (i >= n_jobs) break;
        job(i);
        if (done_count.fetch_add(1, std::memory_order_acq_rel) + 1 == n_jobs) {
            std::lock_guard<std::mutex> dlk(mu);
            cv_done.notify_one();
        }
    }

    // Wait for any workers still finishing.
    std::unique_lock<std::mutex> lk(mu);
    cv_done.wait(lk, [this]() {
        return done_count.load(std::memory_order_acquire) >= n_jobs;
    });
    n_jobs = 0;
    job    = nullptr;
}

llama_zstd_ctx_state::~llama_zstd_ctx_state() {
    for (auto & [name, zs] : seekables) {
        if (zs) {
            ZSTD_seekable_free(zs);
        }
    }
    // Clear tensor->extra pointers we set, to avoid dangling references.
    for (struct ggml_tensor * t : streaming_tensors) {
        t->extra = nullptr;
    }
    // Delete the per-tensor streaming traits objects we own.
    for (void * p : streaming_traits_ptrs) {
        delete static_cast<zstd_streaming_traits *>(p);
    }
}

llama_zstd_ctx_state * llama_zstd_ctx_init(
        const llama_zstd_model_state & model_state,
        const struct llama_model     & model) {
    if (model_state.weights.empty()) {
        return nullptr;
    }

    auto * ctx = new llama_zstd_ctx_state();

    // Hot cache sized to bytes_saved, rounded up to nearest 256 KB frame.
    // Override via LLAMA_ZSTD_CACHE_MB env var: 0 disables caching entirely
    // (every access re-decompresses), large values let the cache hold all
    // decompressed weights indefinitely (fastest, no memory savings).
    static const size_t FRAME_BYTES = 256 * 1024;
    size_t saved = model_state.bytes_before > model_state.bytes_after
            ? model_state.bytes_before - model_state.bytes_after : 0;
    size_t default_sz = (saved + FRAME_BYTES - 1) & ~(FRAME_BYTES - 1);
    const char * env_mb = std::getenv("LLAMA_ZSTD_CACHE_MB");
    if (env_mb != nullptr) {
        long long v = atoll(env_mb);
        if (v < 0) v = 0;
        ctx->cache.max_bytes = (size_t)v * 1024 * 1024;
        LLAMA_LOG_INFO("llama_zstd_ctx_init: hot cache = %.1f MB (LLAMA_ZSTD_CACHE_MB override; default would have been %.1f MB)\n",
                       ctx->cache.max_bytes / 1e6, default_sz / 1e6);
    } else {
        ctx->cache.max_bytes = 0;
        LLAMA_LOG_INFO("llama_zstd_ctx_init: hot cache = 0 MB (default; set LLAMA_ZSTD_CACHE_MB to enable; bytes_saved = %.1f MB)\n",
                       saved / 1e6);
    }

    // Build name -> tensor* lookup from the model.
    std::unordered_map<std::string, struct ggml_tensor *> tensor_by_name;
    for (const auto & [name, t] : model.tensors_by_name) {
        tensor_by_name[name] = t;
    }

    int n_streaming = 0;

    for (const auto & [name, tc] : model_state.weights) {
        if (tc.original_size == 0) {
            ctx->seekables[name] = nullptr;
            continue;
        }

        ZSTD_seekable * zs = ZSTD_seekable_create();
        if (!zs) {
            delete ctx;
            throw std::runtime_error("llama_zstd_ctx_init: ZSTD_seekable_create failed");
        }
        size_t r = ZSTD_seekable_initBuff(zs, tc.data.data(), tc.data.size());
        if (ZSTD_isError(r)) {
            ZSTD_seekable_free(zs);
            delete ctx;
            throw std::runtime_error(std::string("llama_zstd_ctx_init: initBuff: ") + ZSTD_getErrorName(r));
        }
        ctx->seekables[name] = zs;

        // Register streaming traits on F16 tensors so MUL_MAT (2D weights)
        // and MUL_MAT_ID (3D MoE expert weights) are handled by the
        // cache-resident decompression pipeline.
        auto tit = tensor_by_name.find(name);
        if (tit != tensor_by_name.end()
                && tit->second->type == GGML_TYPE_F16
                && ggml_is_contiguous(tit->second)
                && tit->second->ne[3] == 1) {
            const int64_t ne00 = tit->second->ne[0];
            const int64_t ne01 = tit->second->ne[1];
            const int64_t ne02 = tit->second->ne[2];
            const size_t  row_bytes = (size_t)ne00 * sizeof(ggml_fp16_t);
            const size_t  nb02      = (size_t)ne01 * row_bytes;
            const bool    is_3d     = (ne02 > 1);

            // Pre-validate frame alignment: every full-size frame must hold an
            // integer number of weight rows so the streaming loop can attribute
            // them cleanly.  The last frame may be shorter; we handle that.
            const unsigned n_frames = ZSTD_seekable_getNumFrames(zs);
            bool aligned = true;
            if (n_frames > 1) {
                size_t f0_sz = ZSTD_seekable_getFrameDecompressedSize(zs, 0);
                if (f0_sz % row_bytes != 0) {
                    aligned = false;
                }
            }

            if (aligned) {
                auto * traits      = new zstd_streaming_traits();
                traits->tc         = &tc;
                traits->row_bytes  = row_bytes;
                traits->ne00       = ne00;
                traits->ne01       = ne01;
                traits->ne02       = ne02;
                traits->nb02       = nb02;
                traits->aligned    = true;
                traits->is_3d      = is_3d;
                tit->second->extra = traits;
                ctx->streaming_traits_ptrs.push_back(traits);
                ctx->streaming_tensors.push_back(tit->second);
                ++n_streaming;
            }
        }
    }

    // Register the global extra_buffer_type once so ggml_cpu_extra_compute_forward
    // will call our compute_forward for intercepted MUL_MAT nodes.
    if (n_streaming > 0) {
        static std::once_flag s_registered;
        std::call_once(s_registered, []() {
            ggml_backend_cpu_get_extra_buffer_types().push_back(&g_zstd_buft);
        });
        LLAMA_LOG_INFO("llama_zstd_ctx_init: %d tensors registered for streaming MUL_MAT\n",
                       n_streaming);
    }

    // Persistent worker pool for parallel LRU-path decompression.  Size it to
    // hardware_concurrency - 1 so caller thread participation brings us to full
    // core count without oversubscription.
    unsigned n_pool = std::thread::hardware_concurrency();
    if (n_pool > 1) n_pool -= 1;
    if (n_pool > 0) {
        ctx->pool.start(n_pool);
        LLAMA_LOG_INFO("llama_zstd_ctx_init: decompress pool = %u workers\n", n_pool);
    }

    {
        rss_stats rss = read_rss();
        LLAMA_LOG_INFO("llama_zstd_ctx_init: rss=%zu MB peak=%zu MB\n", rss.cur_mb, rss.peak_mb);
    }

    return ctx;
}

// ---------------------------------------------------------------------------
// Partial decompression helpers
// ---------------------------------------------------------------------------

// Ensure the frames covering [byte_start, byte_end) are decompressed in `pe`.
// Only touches frames not already present.
static void partial_ensure_range(
        llama_zstd_partial_entry & pe,
        ZSTD_seekable            * zs,
        size_t                     byte_start,
        size_t                     byte_end) {

    if (byte_end > pe.original_size) byte_end = pe.original_size;
    if (byte_start >= byte_end)      return;

    unsigned frame_lo = ZSTD_seekable_offsetToFrameIndex(zs, (unsigned long long)byte_start);
    unsigned frame_hi = ZSTD_seekable_offsetToFrameIndex(zs, (unsigned long long)(byte_end - 1));

    for (unsigned f = frame_lo; f <= frame_hi; ++f) {
        if (pe.frame_present[f]) continue;

        unsigned long long off = ZSTD_seekable_getFrameDecompressedOffset(zs, f);
        size_t             fsz = ZSTD_seekable_getFrameDecompressedSize(zs, f);

        size_t r = ZSTD_seekable_decompress(zs, pe.buf.data() + off, fsz, off);
        if (ZSTD_isError(r)) {
            LLAMA_LOG_ERROR("partial_ensure_range: frame %u: %s\n", f, ZSTD_getErrorName(r));
            continue;
        }
        pe.frame_present[f]    = true;
        pe.decompressed_bytes += fsz;
    }
}

// Collect unique row indices from a GET_ROWS index tensor (I32).
// Returns sorted, deduplicated indices.
static std::vector<int32_t> collect_row_indices(const struct ggml_tensor * idx) {
    const int32_t * data = (const int32_t *)idx->data;
    int64_t n = ggml_nelements(idx);
    std::vector<int32_t> rows(data, data + n);
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}

// ---------------------------------------------------------------------------
// Graph-level pre-compute decompressor
// ---------------------------------------------------------------------------

void llama_zstd_decompress_graph(
        struct ggml_cgraph          * gf,
        llama_compressed_weight_map & weights,
        llama_zstd_ctx_state        & ctx) {

    static std::atomic<uint64_t> s_dg_call{0};
    const uint64_t dg_call = s_dg_call.fetch_add(1, std::memory_order_relaxed);
    const int64_t  t0_dg   = ggml_time_us();
    int n_dg_hit = 0, n_dg_miss = 0;

    const bool no_cache = (ctx.cache.max_bytes == 0);

    // Free last pass's temporary decompressed buffers (no-cache mode).
    ctx.nocache_bufs.clear();

    // Clear preloaded on every streaming tensor so that an LRU eviction since
    // the last pass is detected and compute_forward falls back to streaming decomp.
    for (struct ggml_tensor * t : ctx.streaming_tensors) {
        static_cast<zstd_streaming_traits *>(t->extra)->preloaded = nullptr;
    }

    // Pass 1: classify every compressed tensor in this graph.
    //   - Track which ops use each tensor as src[0].
    //   - A tensor is "partial-eligible" if it is ONLY used as src[0] of
    //     GET_ROWS nodes and every corresponding src[1] has valid data.
    //   - All other compressed tensors go through full decompression.

    struct tensor_usage {
        bool all_get_rows   = true;  // every use is GET_ROWS src[0]
        bool indices_ready  = true;  // every GET_ROWS src[1]->data != nullptr
        bool all_mul_mat_s0 = true;  // every use is MUL_MAT src[0] with streaming traits
        bool has_streaming  = false; // first sighting had src->extra != nullptr
        // Collected GET_ROWS index tensors (src[1]) for this weight.
        std::vector<const struct ggml_tensor *> idx_tensors;
    };

    std::unordered_map<std::string, tensor_usage> usage;
    // graph_tensors preserves first-encounter order so Pass 2a processes tensors
    // in the same order the graph will use them — favours keeping early tensors
    // in cache over late ones when eviction is needed.
    std::vector<std::string>        graph_tensors;
    std::unordered_set<std::string> graph_tensor_set; // dedup helper

    for (int n = 0; n < ggml_graph_n_nodes(gf); ++n) {
        struct ggml_tensor * node = ggml_graph_node(gf, n);
        if (!node) continue;

        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            struct ggml_tensor * src = node->src[i];
            if (!src || src->name[0] == '\0') continue;
            if (!weights.count(src->name))    continue;

            if (graph_tensor_set.insert(src->name).second) {
                graph_tensors.push_back(src->name);
            }
            auto & u = usage[src->name];

            if (src->extra != nullptr) {
                u.has_streaming = true;
            }

            if (i == 0 && node->op == GGML_OP_GET_ROWS) {
                // This tensor is used as the data source for GET_ROWS.
                struct ggml_tensor * idx = node->src[1];
                if (idx && idx->data && idx->type == GGML_TYPE_I32) {
                    u.idx_tensors.push_back(idx);
                } else {
                    u.indices_ready = false;
                }
                u.all_mul_mat_s0 = false;
            } else if (i == 0
                    && (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID)
                    && src->extra != nullptr) {
                u.all_get_rows = false;
                // all_mul_mat_s0 stays true for streaming-eligible MUL_MAT(_ID) use.
            } else {
                // Used in some other position or op — needs full tensor.
                u.all_get_rows   = false;
                u.all_mul_mat_s0 = false;
            }
        }
    }

    // Pass 2: decompress.
    //
    // Structure: serial plan -> parallel decompress -> serial commit.
    //
    // The decompression itself (ZSTD_seekable_decompress) is by far the most
    // expensive per-tensor work.  Each tensor uses its own seekable handle,
    // so running ZSTD calls on different tensors in parallel has no contention.
    // LRU cache and ctx.partial map mutations are kept in the serial phases.
    //
    // This is especially important for MoE models (--cpu-moe): expert weight
    // tensors are 3D and fail the streaming-registration filter, so all of
    // them land in the full-decompress path.  Without parallelism, thread 0
    // serially decompresses dozens of expert tensors per forward pass while
    // every other core sits idle waiting for the compute phase.

    std::unordered_map<std::string, uint8_t *> ready;
    int n_dg_stream = 0;

    struct DecompressJob {
        std::string            name;
        ZSTD_seekable        * zs = nullptr;
        llama_tensor_compressed * tc = nullptr;

        enum Kind {
            FULL_FRESH,            // decompress whole tensor into buf, commit to cache
            PROMOTE_FROM_PARTIAL,  // fill missing frames in pe.buf, then move to cache
            PARTIAL_ENSURE,        // ensure given byte ranges exist in pe, keep in partial
        } kind;

        // FULL_FRESH: pre-allocated by planning phase (from cache.alloc).
        std::vector<uint8_t> buf;

        // PROMOTE_FROM_PARTIAL / PARTIAL_ENSURE: pointer into ctx.partial.
        llama_zstd_partial_entry * pe = nullptr;

        // PARTIAL_ENSURE: byte ranges to ensure decompressed.
        std::vector<std::pair<size_t, size_t>> ranges;
    };

    std::vector<DecompressJob> jobs;
    jobs.reserve(graph_tensors.size());

    // live: entries already committed (or reserved) for this pass — only these
    // are protected from eviction.  Built incrementally so early tensors can
    // evict late-pass cache entries rather than pinning everything upfront.
    std::unordered_set<std::string> live;

    // --- Pass 2a: planning (serial) ---
    for (const std::string & name : graph_tensors) {
        llama_tensor_compressed & tc = weights.at(name);

        // Uncompressed (poor ratio) — pointer directly into compressed store.
        if (tc.original_size == 0) {
            ready[name] = tc.data.data();
            continue;
        }

        auto uit = usage.find(name);
        bool partial_ok = uit != usage.end()
                       && uit->second.all_get_rows
                       && uit->second.indices_ready
                       && !uit->second.idx_tensors.empty();

        // Streaming-only path: in no-cache mode, when the tensor is registered
        // with streaming traits and is only ever the src[0] of MUL_MAT,
        // compute_forward decompresses frame-by-frame into an L2-resident
        // buffer.  Skipping pre-decompression here is what actually saves the
        // ~12 GB of decompressed weights from materialising in RAM.
        bool stream_only = no_cache
                        && uit != usage.end()
                        && uit->second.has_streaming
                        && uit->second.all_mul_mat_s0;
        if (stream_only) {
            ++n_dg_stream;
            // Leave name out of `ready` so Pass 3 doesn't patch src->data;
            // `preloaded` was cleared at the top of this function so
            // compute_forward will take the streaming path.
            continue;
        }

        ZSTD_seekable * zs = ctx.seekables.at(name);

        if (partial_ok) {
            // Partial-eligible: ensure/extend ctx.partial entry, ready[] points
            // directly into it (no commit to LRU cache).
            unsigned n_frames = ZSTD_seekable_getNumFrames(zs);
            auto pit = ctx.partial.find(name);
            if (pit == ctx.partial.end()) {
                llama_zstd_partial_entry pe;
                pe.original_size      = tc.original_size;
                pe.decompressed_bytes = 0;
                pe.buf.resize(tc.original_size);
                pe.frame_present.assign(n_frames, false);
                pit = ctx.partial.emplace(name, std::move(pe)).first;
            }

            DecompressJob job;
            job.name = name;
            job.zs   = zs;
            job.tc   = &tc;
            job.kind = DecompressJob::PARTIAL_ENSURE;
            job.pe   = &pit->second;

            size_t row_stride = tc.nb[1];
            for (const struct ggml_tensor * idx : uit->second.idx_tensors) {
                auto rows = collect_row_indices(idx);
                for (int32_t r : rows) {
                    size_t row_start = (size_t)r * row_stride;
                    size_t row_end   = row_start + row_stride;
                    job.ranges.emplace_back(row_start, row_end);
                }
            }
            jobs.push_back(std::move(job));

            // Promote any prior cache entry to MRU (no-op if absent).
            ctx.cache.get(name);
        } else {
            // Full decompression path.
            auto pit = ctx.partial.find(name);
            if (!no_cache && pit != ctx.partial.end()) {
                // Promote partial -> full: fill missing frames in pe.buf,
                // then commit the completed buffer to the LRU cache.
                DecompressJob job;
                job.name = name;
                job.zs   = zs;
                job.tc   = &tc;
                job.kind = DecompressJob::PROMOTE_FROM_PARTIAL;
                job.pe   = &pit->second;
                jobs.push_back(std::move(job));
                // Reserve space now (serial, before parallel phase).
                // Pass only entries already committed/reserved this pass so that
                // tensors not yet seen can be evicted by earlier ones.
                ctx.cache.alloc(tc.original_size, &live);
                live.insert(name); // protect this slot from subsequent allocs
            } else {
                if (!no_cache) {
                    uint8_t * ptr = ctx.cache.get(name); // cache hit promotes to MRU
                    if (ptr) {
                        ready[name] = ptr;
                        live.insert(name); // already in cache — protect from eviction
                        ++n_dg_hit;
                        continue;
                    }
                }
                ++n_dg_miss;
                DecompressJob job;
                job.name = name;
                job.zs   = zs;
                job.tc   = &tc;
                job.kind = DecompressJob::FULL_FRESH;
                job.buf  = no_cache ? std::vector<uint8_t>(tc.original_size)
                                    : ctx.cache.alloc(tc.original_size, &live);
                live.insert(name); // protect this pre-allocated slot from subsequent allocs
                jobs.push_back(std::move(job));
            }
        }
    }

    // --- Pass 2b: parallel decompress (via persistent pool) ---
    const int64_t t0_decomp = ggml_time_us();
    if (!jobs.empty()) {
        ctx.pool.run(jobs.size(), [&jobs](size_t i) {
            DecompressJob & job = jobs[i];
            switch (job.kind) {
                case DecompressJob::FULL_FRESH: {
                    size_t r = ZSTD_seekable_decompress(
                        job.zs, job.buf.data(), job.tc->original_size, 0);
                    if (ZSTD_isError(r)) {
                        LLAMA_LOG_ERROR("zstd decompress %s: %s\n",
                            job.name.c_str(), ZSTD_getErrorName(r));
                    }
                    break;
                }
                case DecompressJob::PROMOTE_FROM_PARTIAL: {
                    unsigned n_fr = ZSTD_seekable_getNumFrames(job.zs);
                    llama_zstd_partial_entry & pe = *job.pe;
                    for (unsigned f = 0; f < n_fr; ++f) {
                        if (pe.frame_present[f]) continue;
                        unsigned long long off = ZSTD_seekable_getFrameDecompressedOffset(job.zs, f);
                        size_t             fsz = ZSTD_seekable_getFrameDecompressedSize(job.zs, f);
                        size_t r = ZSTD_seekable_decompress(job.zs, pe.buf.data() + off, fsz, off);
                        if (ZSTD_isError(r)) {
                            LLAMA_LOG_ERROR("zstd promote %s frame %u: %s\n",
                                job.name.c_str(), f, ZSTD_getErrorName(r));
                            continue;
                        }
                        pe.frame_present[f]    = true;
                        pe.decompressed_bytes += fsz;
                    }
                    break;
                }
                case DecompressJob::PARTIAL_ENSURE: {
                    for (auto & rng : job.ranges) {
                        partial_ensure_range(*job.pe, job.zs, rng.first, rng.second);
                    }
                    break;
                }
            }
        });
    }

    // --- Pass 2c: commit (serial) ---
    for (DecompressJob & job : jobs) {
        switch (job.kind) {
            case DecompressJob::FULL_FRESH: {
                uint8_t * ptr;
                if (no_cache) {
                    ctx.nocache_bufs.push_back(std::move(job.buf));
                    ptr = ctx.nocache_bufs.back().data();
                } else {
                    ptr = ctx.cache.commit(job.name, std::move(job.buf));
                }
                ready[job.name] = ptr;
                break;
            }
            case DecompressJob::PROMOTE_FROM_PARTIAL: {
                // no_cache skips the partial promotion path entirely in Pass 2a,
                // so this branch is only reached when caching is enabled.
                auto pit = ctx.partial.find(job.name);
                uint8_t * ptr = ctx.cache.commit(job.name, std::move(pit->second.buf));
                ready[job.name] = ptr;
                ctx.partial.erase(pit);
                break;
            }
            case DecompressJob::PARTIAL_ENSURE: {
                ready[job.name] = job.pe->buf.data();
                break;
            }
        }
    }

    const int64_t t1_decomp = ggml_time_us();

    // Pass 3: patch src->data now that all pointers are stable.
    // For streaming tensors (extra != nullptr) also record the preloaded pointer
    // so compute_forward can detect it and let GGML's fast kernel handle the op.
    for (int n = 0; n < ggml_graph_n_nodes(gf); ++n) {
        struct ggml_tensor * node = ggml_graph_node(gf, n);
        if (!node) continue;
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            struct ggml_tensor * src = node->src[i];
            if (!src || src->name[0] == '\0') continue;
            auto it = ready.find(src->name);
            if (it != ready.end()) {
                src->data = it->second;
                if (src->extra != nullptr) {
                    static_cast<zstd_streaming_traits *>(src->extra)->preloaded = it->second;
                }
            }
        }
    }

    const int64_t t1_dg = ggml_time_us();
    // Log first 5 calls (warmup) then every 50 (steady state).
    if (dg_call < 5 || dg_call % 50 == 0) {
        rss_stats rss = read_rss();
        LLAMA_LOG_INFO("zstd decompress_graph #%zu: "
                       "hit=%d miss=%d stream=%d jobs=%zu | "
                       "decomp=%.2fms total=%.2fms | rss=%zu MB peak=%zu MB\n",
                       (size_t)dg_call + 1,
                       n_dg_hit, n_dg_miss, n_dg_stream, jobs.size(),
                       (t1_decomp - t0_decomp) / 1e3f,
                       (t1_dg - t0_dg) / 1e3f,
                       rss.cur_mb, rss.peak_mb);
    }
}

#endif // GGML_USE_ZSTD
