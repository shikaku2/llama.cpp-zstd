# CLAUDE.md — `--cpu-weight-zstd` / `--igpu-weight-zstd` Feature Spec

## Goal

Implement transparent zstd compression for model weight tensors stored in CPU RAM and/or
GPU-accessible unified memory. Two flags:

- `--cpu-weight-zstd [1-19]` — compresses CPU-backend tensors, decompresses on CPU before compute
- `--igpu-weight-zstd [1-19]` — umbrella flag: enables CPU compression AND GPU-side compression
  for iGPU/unified memory setups where GPU and CPU share the same DDR5 bandwidth pool

Both increase effective memory bandwidth without changing model quality.

---

## Motivation

- CPU RAM / unified memory bandwidth is the primary bottleneck for weight-bound inference
- Tensors are read every forward pass but written only once (at load time)
- zstd level 3 decompression sustains ~1-3 GB/s per CPU thread
- BF16/FP8 weights compress 20-25%; quantized weights compress poorly and are skipped
  automatically via per-type threshold sampling
- **iGPU case**: GPU and CPU share the same DDR5 pool (780M, XDNA, Apple Silicon, etc.).
  GPU-assigned tensors hit the same bandwidth wall as CPU tensors. Compressing them in
  shared RAM and decompressing before GPU compute yields the same effective bandwidth gain
  as the CPU path — without any PCIe transfer involved.
- **Discrete GPU case**: GPU has its own GDDR6/HBM — `--igpu-weight-zstd` GPU path provides no
  benefit and should no-op with a warning if discrete VRAM is detected.

---

## Scope

### `--cpu-weight-zstd`
- Compresses any tensor assigned to the CPU backend at load time
- Decompression on CPU cores immediately before compute
- Covers: full CPU inference, partial GPU offload layers, `--cpu-moe`, any combination

### `--igpu-weight-zstd`
- Implies `--cpu-weight-zstd` behavior for all CPU-backend tensors
- Additionally compresses GPU-backend tensors when unified/shared memory is detected
- Decompression for GPU tensors via compute shader (see GPU Decompression section)
- No-ops the GPU path with a warning if discrete VRAM is detected
- Does not affect KV cache, activations, or intermediate buffers in either path

---

## Dependencies

Add libzstd as an optional dependency:

```cmake
# CMakeLists.txt
option(LLAMA_ZSTD "Enable zstd compression for --cpu-weight-zstd / --igpu-weight-zstd" OFF)

if (LLAMA_ZSTD)
    find_package(zstd REQUIRED)
    target_link_libraries(llama PRIVATE zstd::libzstd_static)
    target_compile_definitions(llama PRIVATE GGML_USE_ZSTD)
endif()
```

---

## CLI Flags

```c
// common/arg.cpp
{ "--cpu-weight-zstd", "[1-19]",
  "compress model weight tensors in CPU RAM with zstd to increase effective bandwidth.\n"
  "accepts an optional compression level 1-19 (default: 3).\n"
  "  level 1  : fastest compression, minimal load time overhead\n"
  "  level 3  : default - good ratio, reasonable load time (recommended)\n"
  "  level 19 : maximum ratio, slow to compress (decompress speed unchanged)\n"
  "tensors that do not compress well are stored raw automatically.\n"
  "most useful for BF16/FP8 models and VRAM-constrained offload scenarios.",
  [](common_params & params, const std::string & value) {
      params.cpu_weight_zstd_level = value.empty() ? 3 : std::stoi(value);
      if (params.cpu_weight_zstd_level < 1 || params.cpu_weight_zstd_level > 19) {
          throw std::invalid_argument("--cpu-weight-zstd level must be 1-19");
      }
  }
},
{ "--igpu-weight-zstd", "[1-19]",
  "compress all model weight tensors (CPU and GPU) with zstd.\n"
  "umbrella flag: implies --cpu-weight-zstd and additionally compresses GPU-backend\n"
  "tensors for iGPU/unified memory setups (780M, XDNA, Apple Silicon, etc.) where\n"
  "GPU and CPU share the same DDR5 bandwidth pool.\n"
  "GPU path is a no-op with a warning if discrete VRAM is detected.\n"
  "same level argument as --cpu-weight-zstd (default: 3).",
  [](common_params & params, const std::string & value) {
      int level = value.empty() ? 3 : std::stoi(value);
      if (level < 1 || level > 19) {
          throw std::invalid_argument("--igpu-weight-zstd level must be 1-19");
      }
      params.cpu_weight_zstd_level = level;
      params.igpu_weight_zstd_level = level;
  }
},
{ "--cpu-weight-zstd-threshold", "FLOAT",
  "per-type compression ratio threshold for --cpu-weight-zstd / --igpu-weight-zstd (default: 0.90).\n"
  "tensor types that do not compress below this ratio are stored raw.\n"
  "set to 1.0 to compress all types regardless of ratio.\n"
  "applies to both CPU and GPU paths.",
  [](common_params & params, const std::string & value) {
      params.cpu_weight_zstd_threshold = std::stof(value);
  }
},
```

```c
// common/common.h
struct common_params {
    // ... existing fields ...
    int   cpu_weight_zstd_level     = 0;     // 0 = disabled, 1-19 = enabled at that level
    int   igpu_weight_zstd_level     = 0;     // 0 = disabled, 1-19 = enabled (unified mem only)
    float cpu_weight_zstd_threshold = 0.90f; // shared by both paths
    bool  cpu_weight_zstd_validate  = false;
};
```

---

## Data Structures

```c
// llama-impl.h or llama-model.h

struct llama_tensor_compressed {
    std::vector<uint8_t>  data;          // compressed bytes (or raw if sentinel)
    size_t                original_size; // 0 = sentinel: stored raw, skip decompression
    ggml_type             type;
    std::vector<int64_t>  shape;         // original tensor shape {ne[0..3]}
};

// Map from tensor name to compressed storage
// Populated at load time for all CPU-backend tensors
using llama_compressed_weight_map = std::unordered_map<std::string, llama_tensor_compressed>;
```

---

## Compression — Model Load Path

### Single Pass: Try Every Tensor

No separate dry run phase. Every eligible tensor is compressed individually. If the result
meets the threshold it is kept compressed; if not, it is stored raw. This catches
surprisingly compressible tensors that per-type sampling would miss — a Q5_K expert tensor
might compress poorly while a Q5_K norm tensor in the same model compresses well due to
different value distributions.

```c
#ifdef GGML_USE_ZSTD

static llama_tensor_compressed compress_tensor(
    const ggml_tensor * t,
    float               threshold,
    int                 level)
{
    llama_tensor_compressed out;
    out.type          = t->type;
    out.original_size = ggml_nbytes(t);
    out.shape         = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };

    size_t bound = ZSTD_compressBound(out.original_size);
    out.data.resize(bound);

    size_t compressed_size = ZSTD_compress(
        out.data.data(), bound,
        t->data, out.original_size,
        level  // user-specified level (default 3); decompression speed is invariant to level
    );

    if (ZSTD_isError(compressed_size)) {
        // Compression failed — store raw
        out.data.resize(out.original_size);
        memcpy(out.data.data(), t->data, out.original_size);
        out.original_size = 0;  // sentinel: stored raw
        return out;
    }

    float ratio = (float)compressed_size / (float)out.original_size;
    if (ratio >= threshold) {
        // Did not compress well enough — discard compressed result, store raw
        out.data.resize(out.original_size);
        memcpy(out.data.data(), t->data, out.original_size);
        out.original_size = 0;  // sentinel: stored raw
        return out;
    }

    // Compression worth keeping
    out.data.resize(compressed_size);
    out.data.shrink_to_fit();
    return out;
}

#endif
```

Abort only if nothing at all compressed — i.e. every single tensor fell back to raw:

```c
float threshold = params.cpu_weight_zstd_threshold;
int   level     = params.cpu_weight_zstd_level;

// Run parallel compression pass — see Threading section
// ...

if (total_compressed == total_original) {
    LLAMA_LOG_WARN(
        "%s: cpu-weight-zstd: no tensors compressed below threshold %.0f%% — disabling\n"
        "         (override with --cpu-weight-zstd-threshold 1.0 to force)\n",
        __func__, threshold * 100.0f);
    params.cpu_weight_zstd_level = 0;
    // compressed_map will be all raw sentinels — safe to proceed but wasteful
    // caller should check and skip the map entirely
    return;
}
```

Parallel compression pass respecting `-t`:

```c
int n_threads = params.cpuparams.n_threads;
// Verify against llama.cpp's cpu_get_num_physical_cores() default and match it if unset

std::vector<std::future<llama_tensor_compressed>> futures;
size_t total_original = 0, total_compressed = 0;
size_t n_kept = 0, n_raw = 0;

for (auto * t : all_cpu_tensors) {
    futures.push_back(std::async(std::launch::async,
        compress_tensor, t, threshold, level));
    if ((int)futures.size() >= n_threads) {
        for (auto & f : futures) {
            auto c = f.get();
            size_t orig = c.original_size ? c.original_size : c.data.size();
            total_original   += orig;
            total_compressed += c.data.size();
            if (c.original_size) n_kept++; else n_raw++;
            compressed_map[tensor_name(t)] = std::move(c);
        }
        futures.clear();
    }
}
for (auto & f : futures) { /* drain remainder */ }

LLAMA_LOG_INFO(
    "%s: cpu-weight-zstd: %.2f GiB -> %.2f GiB (%.1f%% of original) "
    "| %zu tensors compressed, %zu stored raw\n",
    __func__,
    total_original   / 1073741824.0,
    total_compressed / 1073741824.0,
    100.0 * total_compressed / total_original,
    n_kept, n_raw);
```

---

## KV Cache Compression (CPU RAM only)

KV cache follows the compute — if weights are fully in VRAM, KV cache is in VRAM too and
this section does not apply. KV cache ends up in CPU RAM only in these scenarios:

- `--no-kv-offload` is passed explicitly
- Insufficient VRAM to fit both weights and KV cache — llama.cpp spills KV to CPU
- Full CPU inference — everything in RAM

This is not the common case. Most partial-offload users will have KV in VRAM alongside
GPU layers. The compression path below activates only when llama.cpp actually allocates
the KV cache to the CPU backend — check `kv_self.cells` backend at runtime, not at flag
parse time. If KV is in VRAM, skip silently with no warning.

When KV cache is allocated in CPU RAM it is eligible for compression under `--cpu-weight-zstd`
or `--igpu-weight-zstd`.

KV cache differs from weights in one critical way: it is written every forward pass, not just
at load time. Compression must happen on every write and decompression on every read. The
threshold check still applies — if KV data doesn't compress usefully, skip it.

### KV Cache Access Pattern

KV cache is structured as per-layer key and value tensors, grown incrementally as context
fills. Each new token appends one slice. Compression strategy:

- Compress per-layer KV blocks rather than per-token slices — amortizes overhead
- Block size should be tunable; default to 256 tokens per block [~70% confident this is
  a reasonable default — verify against actual access patterns at implementation time]
- Completed blocks (not being actively written) are compression candidates
- The current incomplete block at the head stays uncompressed until sealed

```c
struct llama_kv_block_compressed {
    std::vector<uint8_t> k_data;       // compressed K block (or raw if sentinel)
    std::vector<uint8_t> v_data;       // compressed V block (or raw if sentinel)
    size_t k_original_size;            // 0 = sentinel: stored raw
    size_t v_original_size;            // 0 = sentinel: stored raw
    uint32_t token_start;              // first token index in this block
    uint32_t token_count;              // number of tokens in this block
};
```

### Threshold Check for KV Cache

Run a one-time compressibility check on the first completed KV block. K and V may compress
differently — check separately and track independently:

```c
// After first block seals
float k_ratio = try_compress_ratio(first_block.k_data);
float v_ratio = try_compress_ratio(first_block.v_data);

bool compress_k = k_ratio < params.cpu_weight_zstd_threshold;
bool compress_v = v_ratio < params.cpu_weight_zstd_threshold;

LLAMA_LOG_INFO("%s: kv-zstd: K ratio %.2f%% (%s), V ratio %.2f%% (%s)\n",
    __func__,
    k_ratio * 100.0f, compress_k ? "compressing" : "raw",
    v_ratio * 100.0f, compress_v ? "compressing" : "raw");
```

### Note on KV Cache Compressibility

KV cache activations have different structure than weights. Early context positions tend to
have more redundancy; later positions less so. Compressibility may vary significantly by
model and prompt. The per-block threshold check handles this gracefully — blocks that don't
compress fall back to raw transparently.

TurboQuant handles KV cache via lossy quantization; this is a lossless complement for the
case where KV is in CPU RAM and bandwidth is the bottleneck.

---

### Unified Memory Detection

Before compressing GPU-backend tensors, detect whether the GPU uses unified/shared memory:

```c
static bool gpu_is_unified_memory() {
    // Check ggml backend properties for the active GPU backend
    // Unified = GPU and CPU share the same physical memory pool
    // Indicators:
    //   - AMD iGPU (780M, XDNA): ggml_backend_is_cpu() false, but no discrete VRAM reported
    //   - Apple Silicon: always unified
    //   - Intel Arc iGPU: unified
    //   - Discrete GPU (RX 560, RTX etc): NOT unified — skip GPU path

    // Implementation note: query ggml_backend_dev_props() for memory type.
    // If props.memory_type == GGML_BACKEND_MEM_TYPE_UNIFIED return true.
    // This is a placeholder — verify against actual ggml backend API at implementation time.
    return false; // conservative default — implementer must verify
}
```

If discrete VRAM detected, warn and disable GPU path:

```c
if (params.igpu_weight_zstd_level > 0 && !gpu_is_unified_memory()) {
    LLAMA_LOG_WARN(
        "%s: --igpu-weight-zstd: discrete GPU detected — GPU compression path disabled.\n"
        "         GPU has dedicated VRAM; --cpu-weight-zstd path still active.\n",
        __func__);
    params.igpu_weight_zstd_level = 0;
}
```

### GPU-Side Decompression via Compute Shader

For unified memory GPUs, weights are stored compressed in shared RAM. Decompression happens
via a small compute shader before the matrix operation reads the tensor. The decompressed
result writes to a temporary buffer in the same shared memory pool — no copy across a bus.

```glsl
// zstd_decompress.comp — GLSL compute shader (Vulkan backend)
// Simplified — real zstd requires full FSE/Huffman decode implementation
// Consider using a GPU zstd library (e.g. nvcomp-style or custom FSE kernel)

layout(local_size_x = 64) in;
layout(binding = 0) readonly  buffer CompressedData { uint src[]; };
layout(binding = 1) writeonly buffer OutputData     { uint dst[]; };
layout(push_constant) uniform PushConstants {
    uint compressed_size;
    uint original_size;
};

void main() {
    // FSE + LZ77 decode — non-trivial, reference zstd spec RFC 8878
    // Each workgroup handles one zstd block
}
```

**Practical note**: A full GPU zstd decompressor is non-trivial silicon/shader work. A
simpler interim approach for unified memory:

1. CPU decompresses the tensor into a shared-memory buffer
2. GPU reads from that buffer directly (no copy needed — same physical RAM)
3. Buffer is pinned/mapped so GPU can access it without transfer

This sacrifices some CPU cores during decode but avoids implementing GPU zstd entirely.
Mark the full shader implementation as a follow-on; ship the CPU-decompress-to-shared-buffer
approach first. [~80% confident this is the pragmatic path for an initial implementation]

---

Hook into the CPU backend compute path. Before any operation that reads a weight tensor,
check if it has a compressed entry and decompress into the pre-allocated working buffer.

```c
static void decompress_into(
    const llama_tensor_compressed & src,
    void                          * dst)
{
    if (src.original_size == 0) {
        // Raw sentinel — stored uncompressed
        memcpy(dst, src.data.data(), src.data.size());
        return;
    }

    size_t result = ZSTD_decompress(
        dst, src.original_size,
        src.data.data(), src.data.size()
    );

    if (ZSTD_isError(result)) {
        LLAMA_LOG_ERROR("%s: zstd decompression failed: %s\n",
            __func__, ZSTD_getErrorName(result));
        GGML_ABORT("cpu-weight-zstd: decompression failed — model state unrecoverable");
    }
}
```

Pre-allocate one working buffer per thread slot sized to the largest compressed tensor:

```c
// At load time — scan all compressed entries for max original_size
size_t max_tensor_bytes = 0;
for (auto & [name, c] : compressed_map) {
    max_tensor_bytes = std::max(max_tensor_bytes, c.original_size);
}

// n_slots = max experts per token for MoE, or 1 for dense models
std::vector<std::vector<uint8_t>> decomp_bufs(n_slots,
    std::vector<uint8_t>(max_tensor_bytes));
```

---

## Threading

Both compression (load time) and decompression (per forward pass) respect
`params.cpuparams.n_threads` from the `-t` flag.

- **Load time**: embarrassingly parallel over all tensors — use all `-t` threads, one-time cost
- **Per forward pass**: parallel over simultaneously needed tensors — each needs its own buffer slot

Do not create a separate thread pool. Use llama.cpp's existing ggml threadpool where accessible,
fall back to `std::async` otherwise.

Note: verify llama.cpp's default thread count against `cpu_get_num_physical_cores()` in
`common/common.cpp` and match it rather than assuming `std::thread::hardware_concurrency()`.

---

## Validation Flag

```c
{ "--cpu-weight-zstd-validate",
  "debug: compress and immediately decompress each tensor at load time, verify byte-for-byte.\n"
  "significant load time overhead — do not use in production.",
  [](common_params & params, const std::string &) {
      params.cpu_weight_zstd_validate = true;
  }
},
```

---

## Expected Performance Impact

Based on empirical compression tests (SmolLM3-3B, zstd -1):

| Format | Ratio  | CPU BW Gain (134 GB/s base) | iGPU BW Gain (same pool) |
|--------|--------|-----------------------------|--------------------------|
| BF16   | ~75%   | ~179 GB/s effective (+34%)  | same — shared DDR5       |
| FP8    | ~80%   | ~168 GB/s effective (+25%)  | same — shared DDR5       |
| GPTQ   | ~90%   | ~149 GB/s effective (+11%)  | same — shared DDR5       |
| Q8_0   | ~96%   | stored raw (fails threshold) | stored raw              |
| Q4_0   | ~100%  | stored raw (fails threshold) | stored raw              |

For iGPU, `--igpu-weight-zstd` effectively increases the available bandwidth for the entire
inference stack — both CPU and GPU are reading less data from the same DDR5 pool.

---

## Files to Modify

| File | Change |
|------|--------|
| `CMakeLists.txt` | Add `LLAMA_ZSTD` option, link libzstd |
| `common/common.h` | Add `cpu_weight_zstd_level`, `igpu_weight_zstd_level`, `cpu_weight_zstd_threshold`, `cpu_weight_zstd_validate` |
| `common/arg.cpp` | Register `--cpu-weight-zstd`, `--igpu-weight-zstd`, `--cpu-weight-zstd-threshold`, `--cpu-weight-zstd-validate` |
| `src/llama.cpp` | Compression at load, CPU decompression hook, unified memory detection |
| `src/llama-model.h` | Add `llama_tensor_compressed`, `llama_compressed_weight_map` |
| `ggml/src/ggml-vulkan.cpp` | GPU decompression shader dispatch (follow-on) |
| `ggml/src/zstd_decompress.comp` | Vulkan compute shader for GPU-side decompress (follow-on) |

---

## Out of Scope

- GPU zstd shader for discrete VRAM (no bandwidth benefit — discrete GPU has own HBM/GDDR6)
- KV cache compression when KV is in VRAM (TurboQuant handles that)
- Compressing activations or intermediate buffers
- Any compression algorithm other than zstd, or maybe lz4 (lz4 doesn't compress nearly as well but worth trying perhaps)
- Runtime recompression or adaptive threshold adjustment
- Compressed weight caching to disk (natural follow-on — eliminates load-time cost on repeat runs)

---

## Notes

- Per-tensor compression (not per-type) is intentional. A Q5_K norm tensor and a Q5_K expert
  tensor may have very different compressibility due to different value distributions. Trying
  every tensor and discarding failures catches outliers that sampling would miss.
- `--igpu-weight-zstd` is the recommended flag for iGPU users. It compresses everything in the
  shared DDR5 pool — CPU and GPU tensors alike — maximizing effective bandwidth for the
  whole inference stack.
- `--cpu-weight-zstd` is the flag for discrete GPU users who are offloading layers to CPU RAM.
- Both flags default to level 3 when given without an argument. Level 19 is for users who
  load the model once and run many tokens — pay the compression cost once, benefit forever.
- Decompression speed is invariant across zstd levels 1-19. Only compression (load time) is affected.
- `int cpu_weight_zstd_level` / `int igpu_weight_zstd_level` — 0 means disabled, 1-19 enabled.
- The GPU shader decompressor is a follow-on. Initial GPU path ships as CPU-decompress into
  a shared-memory buffer that the GPU reads directly — works on unified memory, no bus copy.
- KV cache compression applies only when KV is in CPU RAM. It is lossless and complements
  TurboQuant rather than replacing it — use both if available.
- libzstd is a transitive dependency in most Linux environments. Static linking preferred.
- If nothing compresses below threshold, a warning is logged and the feature disables. No hard fail.
