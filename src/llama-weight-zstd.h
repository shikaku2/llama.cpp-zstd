#pragma once

#ifdef GGML_USE_ZSTD

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <string>
#include <unordered_map>
#include <vector>

// Per-tensor compressed blob.  original_size == 0 means the tensor compressed poorly and is
// stored raw (uncompressed) in data — the pre_node_callback still handles it uniformly.
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

// Context-level state owned by llama_context — one per inference context.
struct llama_zstd_ctx_state {
    // One seekable decompressor per compressed tensor, pointing at model's compressed data.
    std::unordered_map<std::string, ZSTD_seekable_s *> seekables;
    // Working buffers: slot i is used for src[i] of the current node.
    std::vector<std::vector<uint8_t>> decomp_bufs;

    ~llama_zstd_ctx_state();
};

// Bundles model weights map + context state so the callback can reach both.
struct llama_zstd_callback_data {
    const llama_compressed_weight_map * weights; // from model (read-only)
    llama_zstd_ctx_state              * ctx;     // per-context mutable state
};

// Compress CPU-backend tensors in model after load_tensors().
// Populates model.zstd_state.  No-op if params.cpu_weight_zstd_level == 0.
void llama_weight_zstd_compress(struct llama_model & model, const struct llama_model_params & params);

// Create context-level state (seekable decompressors + working buffers) from the model state.
// Returns nullptr if model has no compressed weights.
llama_zstd_ctx_state * llama_zstd_ctx_init(const llama_zstd_model_state & model_state);

// Allocate callback data from pointers to model weights and context state.
llama_zstd_callback_data * llama_zstd_callback_data_init(
        const llama_compressed_weight_map * weights,
        llama_zstd_ctx_state              * ctx);

// The ggml pre_node_callback.  user_data must be llama_zstd_callback_data*.
void llama_zstd_pre_node_cb_impl(struct ggml_tensor * node, void * user_data);

#endif // GGML_USE_ZSTD
