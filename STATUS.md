# Gizmo Development Status

## ✅ Completed

### Core Infrastructure
- [x] Project structure with CMake build system
- [x] CLI parser with all commands wired
- [x] Layer manager (bookkeeping)
- [x] Model manager with GGUF file scanning and header parsing (layers, embd, vocab, arch, parameter count)
- [x] Inference engine interface
- [x] Thread-count tuning (`-t, --threads`) wired through run/bench/validate/server/sweep
- [x] Interactive `chat` command with streaming output, tok/s stats, and chat-template support via `llama_chat_apply_template`
- [x] Interactive chat TUI as the default no-command experience, with model picker and live speed/RSS/HWM footer
- [x] Shared model discovery between chat TUI and server TUI (`model_discovery.hpp/cpp`)
- [x] `--model-path` CLI flag and `GIZMO_MODEL_PATH` environment variable for custom model directories
- [x] HTTP server with `/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/v1/health`, `/v1/`, and `/health`
- [x] `gizmo serve -m <model>` starts the HTTP server with an interactive feature-toggles screen (CORS, host, port, threads, no-evict, verbose) and a live server dashboard
- [x] Validation harness (`gizmo validate`) comparing sharded vs un-sharded logits
- [x] Performance sweep (`gizmo sweep`) across threads, resident layers, and row size

### llama.cpp Integration
- [x] Inference engine rewritten with llama.cpp API
- [x] Model loading with `n_gpu_layers=0` CPU-only path
- [x] Token generation loop
- [x] Sampler chain (top_k, top_p, temperature, dist)
- [x] Context management

### Per-Block Sharded Engine
- [x] Sequential per-block `ggml_cgraph` construction (`tools/sharded_engine/`)
- [x] Residual and KV-cache threading across blocks
- [x] Support for qwen3 full-attention models
- [x] Tail graph with `model.output_s` scaling
- [x] Optional `madvise(MADV_DONTNEED)` weight eviction
- [x] Single-token decode path with full-cache causal mask
- [x] Row-graph chaining (`-K, --row-size`) for qwen3 models

### Tested & Verified
- [x] CLI commands build and run
- [x] `gizmo run` end-to-end on qwen3-family models (e.g., Qwen3-4B)
- [x] Sharded prefill validated against un-sharded `llama_decode` baseline on qwen3 models
  - Single-token prompt: exact parity (max diff 0.0)
  - Multi-token prompt: exact parity (max diff 0.0)
  - With and without `--evict`
- [x] Sharded decode validated against un-sharded `llama_decode` baseline on qwen3 models
  - Per-token argmax and logits match exactly across decode iterations
- [x] qwen3.5-family models use the per-block sharded engine and match the un-sharded `llama_decode` baseline exactly
  - Validated on Qwen3.5-0.8B-Q8_0, Qwen3.5-2B-Q4_K_S, Qwen3.5-4B-Q4_K_M, and Qwen3.5-9B-Q4_K_M
  - Both full-attention and gated-delta-net recurrent blocks pass the built-in `gizmo validate` prompt suite
- [x] qwen3.8-27B uses the per-block sharded engine and matches the un-sharded baseline exactly
  - `gizmo validate` passes all built-in prompts with zero logit diffs
  - `gizmo run`, `gizmo chat`, and `gizmo server` all work with `-r 1` under 2.1 GB peak resident
- [x] qwen2 / qwen2.5 / MoE models fall back to native `llama_decode` and produce correct output
- [x] Measured memory on qwen3-family models
  - Model load RSS with sharding enabled: **~120 MB** (no mmap prefault)
  - Sharded prefill steady-state: **~510 MB**
  - Sharded decode steady-state: **~500 MB**
  - End-to-end HWM: **~630 MB**
- [x] Measured memory on Qwen3.8-27B Q4_K_M with sharding enabled
  - Model load RSS: **~293 MB**
  - `-r 1` generation HWM: **~2.0 GB**
  - `-r 8` validation HWM: **~12.7 GB** (requires a host with more than 16 GB RAM for comfort)

## ⏳ Pending

### Sharded Engine Expansion
- [x] Per-block sharded engine support for **qwen3.5-family** models (full-attention + recurrent hybrid blocks)
- [x] Re-enable sharded path for `LLM_ARCH_QWEN35`
- [x] ISWA KV-cache support for qwen3.5 full-attention blocks
- [x] Recurrent-state threading for gated-delta-net blocks

### Convenience Features
- [x] `--progress` in-place progress indicator for sharded prefill/decode (TTY only, silenced by `--verbose` or non-TTY stdout)
- [x] Default no-command chat TUI with model picker and live speed/RSS/HWM footer
- [x] `gizmo serve` interactive launcher with model picker, profile picker, and feature toggles
- [x] Headless `gizmo serve -m` for production and Docker use
- [x] `--model-path` / `GIZMO_MODEL_PATH` custom model search directories
- [ ] Automatic sweet-spot recommendation from `gizmo sweep` results
- [ ] Built-in model download via libcurl (current `download` delegates to system `curl`/`wget`)
- [ ] Better token-usage reporting in streaming SSE responses

### Production Readiness
- [x] Graceful shutdown via SIGINT/SIGTERM
- [x] Per-request generation timeout with cancellation support
- [x] Bounded generation queue for safe single-user + parallel subagent use
- [x] OpenAI-compatible `/v1/` endpoints (`/v1/models`, `/v1/completions`, `/v1/chat/completions`, `/v1/health`)
- [x] Chat-template-aware formatting in `/v1/chat/completions` via `llama_chat_apply_template`
- [x] CORS preflight support for browser frontends
- [x] Structured JSON request logging to stderr and/or a file
- [x] Concise `model` id in OpenAI responses (basename without `.gguf`)
- [x] Multi-stage `Dockerfile`
- [x] GitHub Actions CI/release workflow
- [x] `install.sh` release installer
- [x] CPack packaging (TGZ + Debian)

### Polish
- [ ] Reduce sharded-engine diagnostic noise further when `--verbose` is off
- [ ] Performance tuning: find the best default resident-layer count per model family

## Current Behavior by Architecture

| Architecture | Sharded Engine | Fallback | Notes |
|-------------|----------------|----------|-------|
| `qwen3`     | ✅ enabled     | native `llama_decode` if disabled | Validated end-to-end |
| `qwen35`    | ✅ enabled     | native `llama_decode` if disabled | Validated end-to-end on 0.8B–27B models |
| `qwen2`     | ❌ disabled    | native `llama_decode` | Architecture not ported |
| `qwen2.5`   | ❌ disabled    | native `llama_decode` | Architecture not ported |
| `qwen3moe`  | ❌ disabled    | native `llama_decode` | MoE not ported |
| `qwen35moe` | ❌ disabled    | native `llama_decode` | MoE not ported |
| `qwen3vlmoe`| ❌ disabled    | native `llama_decode` | MoE/VL not ported |

## Next Steps

1. Validate remaining local model families (`qwen3`, `qwen2`, `qwen2.5`, MoE) with the refreshed CLI matrix.
2. Implement convenience features (sweet-spot sweep, libcurl download, better SSE usage reporting).
3. Performance tuning: find the best default resident-layer count per model family.
4. Add HTTPS / basic-auth support when leaving the local-only scenario.

## Current Build Command

```bash
# Configure and build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run a bounded end-to-end test on a qwen3 model
./build/gizmo run -m /path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
    -p "What is the capital of France?" -r 1 -n 4 --measure-ram
```
