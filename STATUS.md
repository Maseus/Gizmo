# Gizmo Development Status

## ✅ Completed

### Core Infrastructure
- [x] Project structure with CMake build system
- [x] CLI parser with all commands working
- [x] Layer manager (bookkeeping)
- [x] Model manager with GGUF file scanning
- [x] Inference engine interface
- [x] Thread-count tuning (`-t, --threads`) wired through run/bench/validate/server/sweep
- [x] Simple interactive `chat` command with streaming output and tok/s stats

### llama.cpp Integration
- [x] Inference engine rewritten with llama.cpp API
- [x] Model loading with `n_gpu_layers=0` CPU-only path
- [x] Token generation loop
- [x] Sampler chain (top_k, top_p, temperature, dist)
- [x] Context management

### Per-Block Sharded Engine
- [x] Sequential per-block `ggml_cgraph` construction (`tools/sharded_engine/`)
- [x] Residual, KV-cache, and recurrent-state threading across blocks
- [x] Support for hybrid architectures (qwen3 full-attention + gated-delta-net recurrent layers)
- [x] MRoPE position layout matching native `llama_decode`
- [x] Tail graph with `model.output_s` scaling
- [x] Optional `madvise(MADV_DONTNEED)` weight eviction
- [x] CPU weight repack disabled (`use_extra_bufts=false`) to prevent 27B load OOM
- [x] Integration into `InferenceEngine::generate()` as the prefill path

### Tested & Verified
- [x] CLI commands work correctly
- [x] `gizmo run` end-to-end on qwen3.8:27b (`Qwen3.8-27B-Q4_K_M.gguf`)
- [x] Sharded prefill validated against un-sharded `llama_decode` baseline
  - Single-token prompt: exact parity (max diff 0.0)
  - Multi-token prompt: exact parity (max diff 0.0)
  - With and without `--evict`
- [x] Sharded decode validated against un-sharded `llama_decode` baseline
  - Per-token argmax and logits match exactly across decode iterations
- [x] End-to-end generation produces coherent text
  - `"What is the capital of France?"` → `"The capital of …"`
- [x] Measured memory on qwen3.8:27b
  - Sharded prefill steady-state at `-r 1`: **~560 MB**
  - Sharded decode steady-state at `-r 1`: **~1.5 GB**
  - Model load RSS with sharding enabled: **~298 MB** (no mmap prefault)
  - End-to-end peak HWM with sharding enabled: **~2.0 GB**
  - `--no-shard` baseline load RSS: **~7.5 GB**, HWM **~9.2 GB**

## ⏳ Pending

### Integration Testing
- [x] Automated diff harness comparing sharded vs un-sharded logits across a prompt suite (`gizmo validate`)
- [x] Performance regression sweep for resident-layer / row-size / thread-count matrix (`gizmo sweep`)
- [ ] Automatic sweet-spot recommendation from sweep results

### Features to Add
- [ ] Built-in model download via libcurl
- [ ] GGUF header parsing for automatic layer size detection
- [ ] Interactive multi-turn `chat` with chat-template support

## Key Insight: End-to-End Sharded Generation

llama.cpp's `llama_decode` builds one `ggml_cgraph` containing all transformer blocks. Gizmo now bypasses this for **both prefill and decode** with a custom per-block engine. On qwen3.8:27b:

- Model load uses mmap without prefault when sharding is enabled, so initial RSS stays under ~300 MB.
- Prefill RSS drops to sub-1 GB (`-r 1`).
- Decode RSS stays at ~1.5 GB per token (`-r 1`).
- End-to-end HWM is now ~2.0 GB, down from ~9.5 GB.

## Next Steps

1. ✅ **Reduce diagnostic noise**: sharded-engine per-block progress, argmax reports, and initialization details are now quiet by default. Use `--verbose` / `-v` to restore them.
2. ✅ **Automated validation harness**: `gizmo validate` runs a built-in prompt suite against both sharded and un-sharded paths, comparing last-token argmax and logit diffs. Supports JSON output (`--json`), argmax-only mode (`--argmax-only`), and configurable tolerances (`--max-diff`, `--mean-diff`).
3. ✅ **Performance sweep**: `gizmo sweep` captures wall time and peak RSS for a full `generate()` run across a matrix of `-t` (threads), `-r` (resident layers), and `-K` (row size) values. Supports `--json` and configurable prefill/decode sizes.
4. ✅ **Cleanup**: removed the unused `chat_handler.hpp`/`chat/handler.cpp` stub, dropped `-i/--interactive` and `-u/--url` flags, and updated `download` to print a helpful `curl`/`wget` pointer. Sharded engine still only supports qwen3/qwen3.5-family models; non-hybrid/qwen2 models use `--no-shard` fallback.
5. ✅ **Thread-count tuning**: `-t, --threads` sets CPU threads for native `llama_decode` and the sharded backend; sweep iterates over a thread list as the outer loop.
6. ✅ **Simple chat mode**: `gizmo chat` starts an interactive loop that streams assistant tokens, shows tok/s and memory after each turn, and supports `/reset` to clear context. It uses a plain text context (not the model's chat template).

## Known Findings

- `gizmo validate` reports **exact parity** (max diff 0.0) on `qwen3.8:27b` and `Qwen3.5-9B` with default tolerances.
- `Qwen3-8B` shows small but growing logit drift as prompt length increases (argmax still matches). This is flagged as FAIL under the default tolerances and PASS with `--argmax-only`. The sharded engine was primarily validated on qwen3.5-family hybrid models; pure qwen3 attention routing may need a separate look.
- The sharded engine is **not** compatible with qwen2/qwen2.5-family models or MoE variants (`qwen3moe`, `qwen35moe`, `qwen3vlmoe`); Gizmo now detects this at load time and automatically disables the sharded engine, falling back to native `llama_decode`.
- **Ornith1.5:35B** was tested: it reports `general.architecture = qwen35moe`, runs correctly via the auto fallback, and passes `gizmo validate` with 0.0 logit diff. Sharded-engine support for qwen35moe remains future work.

## Current Build Command

```bash
# Configure and build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run a bounded end-to-end test on qwen3.8:27b
./build/gizmo run -m /path/to/Qwen3.8-27B-Q4_K_M.gguf \
    -p "What is the capital of France?" -r 1 -n 4 --measure-ram
```
