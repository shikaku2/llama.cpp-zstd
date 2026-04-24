#ifdef GGML_USE_ZSTD

#include "llama-weight-zstd.h"
#include "llama-model.h"
#include "llama-impl.h"

#include "../vendor/zstd-seekable/zstd_seekable.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <future>
#include <stdexcept>
#include <thread>

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
    const int   frame_kb   = params.cpu_weight_zstd_frame_kb > 0 ? params.cpu_weight_zstd_frame_kb : 256;
    const bool  validate   = params.cpu_weight_zstd_validate;

    LLAMA_LOG_INFO("%s: compressing CPU weights with zstd level=%d threshold=%.2f frame_kb=%d\n",
                   __func__, level, threshold, frame_kb);

    // Collect eligible tensors (CPU host buffers, has data, has a name)
    std::vector<struct ggml_tensor *> candidates;
    for (auto & [name, tensor] : model.tensors_by_name) {
        if (!tensor || !tensor->data || !tensor->buffer) {
            continue;
        }
        if (!ggml_backend_buffer_is_host(tensor->buffer)) {
            continue;
        }
        size_t sz = ggml_nbytes(tensor);
        if (sz == 0) {
            continue;
        }
        candidates.push_back(tensor);
    }

    if (candidates.empty()) {
        LLAMA_LOG_INFO("%s: no eligible CPU tensors found\n", __func__);
        return;
    }

    // Determine how many threads to use (respect -t setting if available, else HW concurrency)
    const int n_threads = std::max(1, (int)std::thread::hardware_concurrency());

    size_t bytes_before = 0;
    size_t bytes_after  = 0;
    int    n_compressed = 0;
    int    n_skipped    = 0;

    // Compress in parallel
    std::vector<std::pair<std::string, std::future<llama_tensor_compressed>>> futures;
    futures.reserve(candidates.size());

    // Simple thread pool using std::async with a semaphore would be cleaner, but for simplicity
    // process in batches of n_threads
    size_t batch_start = 0;
    while (batch_start < candidates.size()) {
        size_t batch_end = std::min(batch_start + (size_t)n_threads, candidates.size());
        for (size_t i = batch_start; i < batch_end; ++i) {
            struct ggml_tensor * t = candidates[i];
            const void * data      = t->data;
            size_t       sz        = ggml_nbytes(t);
            std::string  name      = std::string(t->name);
            ggml_type    type      = t->type;
            int64_t      ne[GGML_MAX_DIMS];
            size_t       nb[GGML_MAX_DIMS];
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                ne[d] = t->ne[d];
                nb[d] = t->nb[d];
            }

            futures.push_back({name, std::async(std::launch::async,
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

        // Collect this batch
        for (auto & [name, fut] : futures) {
            try {
                auto tc = fut.get();
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
        }
        futures.clear();
        batch_start = batch_end;
    }

    state.bytes_before = bytes_before;
    state.bytes_after  = bytes_after;

    LLAMA_LOG_INFO("%s: compressed %d tensors (%.1f MB → %.1f MB, ratio %.2f), skipped %d\n",
                   __func__,
                   n_compressed,
                   bytes_before / 1e6,
                   bytes_after  / 1e6,
                   bytes_before > 0 ? (float)bytes_after / (float)bytes_before : 1.0f,
                   n_skipped);
}

// ---------------------------------------------------------------------------
// Context-level state
// ---------------------------------------------------------------------------

llama_zstd_ctx_state::~llama_zstd_ctx_state() {
    for (auto & [name, zs] : seekables) {
        if (zs) {
            ZSTD_seekable_free(zs);
        }
    }
}

llama_zstd_ctx_state * llama_zstd_ctx_init(const llama_zstd_model_state & model_state) {
    if (model_state.weights.empty()) {
        return nullptr;
    }

    auto * ctx = new llama_zstd_ctx_state();

    // Find max original tensor size to size the working buffers
    size_t max_bytes = 0;
    for (const auto & [name, tc] : model_state.weights) {
        size_t sz = tc.original_size > 0 ? tc.original_size : tc.data.size();
        max_bytes = std::max(max_bytes, sz);
    }

    ctx->decomp_bufs.resize(GGML_MAX_SRC, std::vector<uint8_t>(max_bytes));

    for (const auto & [name, tc] : model_state.weights) {
        if (tc.original_size == 0) {
            // Stored raw — no seekable context needed
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
    }

    return ctx;
}

// ---------------------------------------------------------------------------
// Pre-node callback
// ---------------------------------------------------------------------------

void llama_zstd_pre_node_cb_impl(struct ggml_tensor * node, void * user_data) {
    auto * cbd = static_cast<llama_zstd_callback_data *>(user_data);
    if (!cbd || !cbd->ctx || !cbd->weights) return;

    auto & ctx     = *cbd->ctx;
    auto & weights = *cbd->weights;

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        struct ggml_tensor * src = node->src[i];
        if (!src || src->name[0] == '\0') continue;

        auto it = weights.find(src->name);
        if (it == weights.end()) continue;

        const llama_tensor_compressed & tc = it->second;
        std::vector<uint8_t> & buf = ctx.decomp_bufs[i];

        if (tc.original_size == 0) {
            // Stored raw
            assert(buf.size() >= tc.data.size());
            memcpy(buf.data(), tc.data.data(), tc.data.size());
        } else {
            // Seekable decompress
            ZSTD_seekable * zs = ctx.seekables.at(src->name);
            size_t r = ZSTD_seekable_decompress(zs, buf.data(), tc.original_size, 0);
            if (ZSTD_isError(r)) {
                // Non-fatal: leave old data — will produce wrong results but won't crash
                return;
            }
        }
        src->data = buf.data();
    }
}

// ---------------------------------------------------------------------------
// Public: create callback data and expose the real callback pointer
// ---------------------------------------------------------------------------

llama_zstd_callback_data * llama_zstd_callback_data_init(
        const llama_compressed_weight_map * weights,
        llama_zstd_ctx_state              * ctx) {
    auto * cbd = new llama_zstd_callback_data;
    cbd->weights = weights;
    cbd->ctx     = ctx;
    return cbd;
}

#endif // GGML_USE_ZSTD
